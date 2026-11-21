//! tf-dsv41-rank-reslice TP2_BUNDLE OUT_DIR NODE [--index-only | --check]: node NODE's rank file of the four-node 2D
//! split (rank{NODE}of4) cut from the TP2 bundle's rank file of TP2 rank NODE % 2 (TP2_BUNDLE/rank{r}/rank-cache, its
//! config at TP2_BUNDLE/model), no checkpoint, no GPU, no Python. The TP2 file must be exactly plan.wants' TP2 split
//! (its DSpark and dense blocks read off its keys); the TP4 entries are plan.wants' at node NODE, each a whole TP2
//! entry or a column range of one (reslice.zig). The file is written as tf-dsv41-rank-cache writes it (same index
//! rows, offsets, footer and name), one node a run. --index-only: print the index, write nothing. --check: compare the
//! file already in OUT_DIR with the TP2 file entry by entry (keys, dtypes, shapes, offsets and every byte, trellis
//! tile by tile) and print a report like tf-dsv41-check's; exit 1 on any mismatch.
const std = @import("std");
const dsv41 = @import("dsv41");
const reslice = dsv41.reslice;

const usage = "usage: tf-dsv41-rank-reslice TP2_BUNDLE OUT_DIR NODE [--index-only | --check]\n";

/// The block a dense key belongs to (layers.i: i; mtp.j: layers + j).
fn block(cfg: dsv41.Config, name: []const u8) !usize {
    var it = std.mem.splitScalar(u8, name, '.');
    const head = it.next() orelse return error.BadKey;
    const i = try std.fmt.parseInt(usize, it.next() orelse return error.BadKey, 10);
    if (std.mem.eql(u8, head, "layers")) return i;
    if (std.mem.eql(u8, head, "mtp")) return cfg.layers + i;
    return error.BadKey;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4 or args.len > 5) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const node = std.fmt.parseInt(u32, args[3], 10) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    var index_only = false;
    var check = false;
    if (args.len == 5) {
        if (std.mem.eql(u8, args[4], "--index-only")) {
            index_only = true;
        } else if (std.mem.eql(u8, args[4], "--check")) {
            check = true;
        } else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    const split = dsv41.plan.Split.node(node, 4) catch {
        std.debug.print("NODE is 0..3\n{s}", .{usage});
        return 2;
    };
    const r = split.rank;
    const cfg = try dsv41.Config.read(a, io, try std.fmt.allocPrint(a, "{s}/model", .{args[1]}));
    const src_path = try std.fmt.allocPrint(a, "{s}/rank{d}/rank-cache", .{ args[1], r });
    var src_dir = try std.Io.Dir.cwd().openDir(io, src_path, .{ .iterate = true });
    defer src_dir.close(io);
    const src_name = try dsv41.rank_cache.find(a, io, src_dir, r, 2);
    var tp2 = try dsv41.rank_cache.open(gpa, io, src_dir, src_name);
    defer tp2.deinit();

    // what the TP2 file was built with, from its keys, and that it is exactly plan.wants' TP2 split for that
    var dn: dsv41.plan.Dense = .{};
    var dspark = false;
    const keys2 = tp2.entries.keys();
    for (keys2) |k| {
        if (std.mem.startsWith(u8, k, "mtp.")) dspark = true;
        if (!std.mem.endsWith(u8, k, "|dense")) continue;
        const key = try reslice.Key.parse(k);
        const i = try block(cfg, key.name);
        if (i >= 64) return error.BadKey;
        if (std.mem.endsWith(u8, key.name, ".attn.wo_b")) {
            dn.wo_b |= @as(u64, 1) << @intCast(i);
        } else if (std.mem.endsWith(u8, key.name, ".ffn.shared_experts.w2")) {
            dn.shared_w2 |= @as(u64, 1) << @intCast(i);
        } else return error.BadKey;
    }
    const want2 = try dsv41.plan.wants(a, cfg, .{ .rank = r, .world = 2 }, dspark, dn);
    var same2 = want2.len == keys2.len;
    if (same2) for (want2, keys2) |x, k| {
        if (!std.mem.eql(u8, x.key, k)) same2 = false;
    };
    if (!same2) {
        std.debug.print("{s}/{s} is not plan.wants' TP2 split of rank {d} ({d} entries, plan {d})\n", .{ src_path, src_name, r, keys2.len, want2.len });
        return 1;
    }
    const by_part = try reslice.parts(a, &tp2);
    const want = try dsv41.plan.wants(a, cfg, split, dspark, dn);
    const cuts = try a.alloc(reslice.Cut, want.len);
    var index: std.Io.Writer.Allocating = .init(gpa);
    defer index.deinit();
    const iw = &index.writer;
    try iw.writeByte('[');
    var total: u64 = 0;
    for (want, cuts, 0..) |x, *c, i| {
        c.* = reslice.cut(&tp2, &by_part, x.key) catch |err| {
            std.debug.print("{s}: {t}\n", .{ x.key, err });
            return 1;
        };
        var bad = false;
        if (x.dtype) |d| bad = bad or d != c.dtype;
        if (x.k16) |k| bad = bad or c.dim(0) != k;
        if (x.n16) |n| bad = bad or c.dim(1) != n;
        if (bad) {
            std.debug.print("{s}: the cut's {t} {any} is not the split's\n", .{ x.key, c.dtype, c.shape[0..c.rank] });
            return 1;
        }
        if (i > 0) try iw.writeAll(", ");
        try dsv41.rank_cache.writeRow(iw, x.key, c.dtype, c.shape[0..c.rank], total, c.bytes());
        total += c.bytes();
    }
    try iw.writeByte(']');
    const body = index.written();
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(body, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    var out_buf: [1 << 16]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    if (index_only) {
        try out.interface.writeAll(body);
        try out.interface.flush();
        return 0;
    }

    var src_file = try src_dir.openFile(io, src_name, .{});
    defer src_file.close(io);
    var src_buf: std.ArrayList(u8) = .empty;
    defer src_buf.deinit(gpa);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);

    if (check) {
        var dir = try std.Io.Dir.cwd().openDir(io, args[2], .{ .iterate = true });
        defer dir.close(io);
        const name = try dsv41.rank_cache.find(a, io, dir, node, 4);
        var ix = try dsv41.rank_cache.open(gpa, io, dir, name);
        defer ix.deinit();
        var file = try dir.openFile(io, name, .{});
        defer file.close(io);
        const keys = ix.entries.keys();
        const vals = ix.entries.values();
        var bad: usize = 0;
        var byte_bad: usize = 0;
        var compared: usize = 0;
        var compared_bytes: u64 = 0;
        var at: u64 = 0;
        const n = @min(keys.len, want.len);
        for (0..n) |i| {
            const c = cuts[i];
            const e = vals[i];
            if (!std.mem.eql(u8, keys[i], want[i].key) or e.dtype != c.dtype or e.rank != c.rank or
                !std.mem.eql(usize, e.shape[0..e.rank], c.shape[0..c.rank]) or e.offset != at or e.bytes != c.bytes())
            {
                bad += 1;
                if (bad <= 10) try out.interface.print("mismatch {d}: file {s} {t} {any} at {d} | cut {s} {t} {any} at {d}\n", .{ i, keys[i], e.dtype, e.shape[0..e.rank], e.offset, want[i].key, c.dtype, c.shape[0..c.rank], at });
                at += c.bytes();
                continue;
            }
            at += c.bytes();
            try src_buf.resize(gpa, c.src.bytes);
            if (try src_file.readPositionalAll(io, src_buf.items, c.src.offset) != c.src.bytes) return error.ShortRead;
            try buf.resize(gpa, e.bytes);
            if (try file.readPositionalAll(io, buf.items, e.offset) != e.bytes) return error.ShortRead;
            compared += 1;
            compared_bytes += e.bytes;
            if (!reslice.same(c, src_buf.items, buf.items)) {
                byte_bad += 1;
                if (byte_bad <= 10) try out.interface.print("bytes {d}: {s} differ from {s}\n", .{ i, keys[i], c.from });
            }
        }
        // the name carries the index hash, so a file renamed onto another index fails too
        const want_name = try std.fmt.allocPrint(a, "rank{d}of4-{s}.bin", .{ node, hex[0..20] });
        const name_ok = std.mem.eql(u8, name, want_name);
        const ok = bad == 0 and byte_bad == 0 and keys.len == want.len and ix.data_end == total and name_ok;
        try out.interface.print("{{\"file\": \"{s}\", \"source\": \"{s}\", \"node\": {d}, \"entries\": {d}, \"planned\": {d}, \"bytes\": {d}, \"index_at\": {d}, \"mismatches\": {d}, \"layout_mismatches\": 0, \"compared\": {d}, \"compared_bytes\": {d}, \"byte_mismatches\": {d}, \"name_is_index_hash\": {}, \"ok\": {}}}\n", .{ name, src_name, node, keys.len, want.len, total, ix.data_end, bad, compared, compared_bytes, byte_bad, name_ok, ok });
        try out.interface.flush();
        return if (ok) 0 else 1;
    }

    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, args[2], .{ .open_options = .{ .iterate = true } });
    defer dir.close(io);
    if (dsv41.rank_cache.find(a, io, dir, node, 4)) |old| {
        std.debug.print("{s} already holds {s}: remove it or name an empty OUT_DIR\n", .{ args[2], old });
        return 1;
    } else |err| if (err != error.NoRankCache) return err;
    const partial = try std.fmt.allocPrint(a, ".rank{d}of4.partial", .{node});
    var file = try dir.createFile(io, partial, .{});
    var file_open = true;
    defer if (file_open) file.close(io);
    errdefer dir.deleteFile(io, partial) catch {};
    var wbuf: [1 << 20]u8 = undefined;
    var fw = file.writer(io, &wbuf);
    const w = &fw.interface;
    var at: u64 = 0;
    var report: u64 = 8 << 30;
    for (cuts) |c| {
        try src_buf.resize(gpa, c.src.bytes);
        if (try src_file.readPositionalAll(io, src_buf.items, c.src.offset) != c.src.bytes) return error.ShortRead;
        if (c.whole()) {
            try w.writeAll(src_buf.items);
        } else {
            try buf.resize(gpa, c.bytes());
            try reslice.copy(c, src_buf.items, buf.items);
            try w.writeAll(buf.items);
        }
        at += c.bytes();
        if (at >= report) {
            std.debug.print("node {d} of 4: {d} of {d} MiB\n", .{ node, at >> 20, total >> 20 });
            report += 8 << 30;
        }
    }
    try w.writeAll(body);
    var tail: [16]u8 = undefined;
    std.mem.writeInt(u64, tail[0..8], at, .little);
    @memcpy(tail[8..], dsv41.rank_cache.magic);
    try w.writeAll(&tail);
    try w.flush();
    try file.sync(io);
    file.close(io);
    file_open = false;
    const name = try std.fmt.allocPrint(a, "rank{d}of4-{s}.bin", .{ node, hex[0..20] });
    try dir.rename(partial, dir, name, io);
    try out.interface.print("{{\"file\": \"{s}\", \"source\": \"{s}\", \"entries\": {d}, \"data_bytes\": {d}, \"index_bytes\": {d}, \"rank\": {d}, \"world\": 4, \"dspark\": {}, \"dense_wo_b\": {d}, \"dense_shared_w2\": {d}}}\n", .{ name, src_name, want.len, at, body.len, node, dspark, @popCount(dn.wo_b), @popCount(dn.shared_w2) });
    try out.interface.flush();
    return 0;
}
