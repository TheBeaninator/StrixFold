//! world 4 on the host (zig build test-dsv41-hip-host): the four-node split's exchanges (prompt2d.Two quarters, catRank
//! and pairGather) on four ranks, each a thread over link_fake's in-process collective, their device buffers host memory.
//! ops.copyRows runs on the host through the launch hook (the fake HIP runtime accepts every launch; the hook does
//! copy16's work), so every byte that moves is the code under test's. A correct split leaves node g = 2 p + r holding
//! exactly TP2 rank r's bytes: the quarters TP2's [2, n, N] gather of both ranks' partials, catRank rank r's whole
//! width, pairGather its pair's two parts in TP2 rank order. The pairs' widths differ (as the experts' 5 / 4 blocks
//! and the head's 253 / 252 do), so the padded parts are exercised too.
const std = @import("std");
const gpu = @import("gpu");
const ops_mod = @import("ops.zig");
const prompt = @import("prompt.zig");
const prompt2d = @import("prompt2d.zig");
const link_fake = @import("link_fake.zig");
const coll = @import("coll.zig");

var driver: gpu.Driver = undefined;
const copy16_id: usize = 0x1010;
const strided_id: usize = 0x1020;

fn accept(_: gpu.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: gpu.abi.Stream, _: gpu.abi.Params, _: gpu.abi.Params) callconv(.c) gpu.abi.Result {
    return 0;
}

/// Set on the calling thread by the vgather test. Off: a strided launch is refused, so the unequal-width test still
/// proves those exchanges launch nothing but copy16.
threadlocal var allow_strided: bool = false;
threadlocal var n_copy: u32 = 0;
threadlocal var n_strided: u32 = 0;

/// copy16 on the host: (src, src_ld / 16, dst, dst_ld / 16, words), a row per grid y. Strided
/// (src, dst, outer, middle, inner, source outer/middle, dest outer/middle) only when this thread asked for it:
/// one block an outer group, the glue kernel's byte index.
fn hostCopy(f: gpu.Function, cfg: gpu.Config, args: *const gpu.Args) gpu.Error!void {
    const id = @intFromPtr(f.handle);
    if (id == copy16_id) {
        n_copy += 1;
        const src = args.integer(0).?;
        const sld = args.integer(1).? * 16;
        const dst = args.integer(2).?;
        const dld = args.integer(3).? * 16;
        const bytes = args.integer(4).? * 16;
        for (0..@as(usize, cfg.grid.y)) |row| {
            const s: [*]const u8 = @ptrFromInt(src + row * sld);
            const d: [*]u8 = @ptrFromInt(dst + row * dld);
            @memcpy(d[0..bytes], s[0..bytes]);
        }
        return;
    }
    if (id != strided_id or !allow_strided) return error.Unsupported;
    if (cfg.grid.y != 1 or cfg.block.x != 256) return error.Unsupported;
    const outer = args.integer(2).?;
    if (@as(u64, cfg.grid.x) != outer) return error.Unsupported;
    n_strided += 1;
    const src = args.integer(0).?;
    const dst = args.integer(1).?;
    const middle = args.integer(3).?;
    const inner: usize = @intCast(args.integer(4).?);
    const src_outer = args.integer(5).?;
    const src_middle = args.integer(6).?;
    const dst_outer = args.integer(7).?;
    const dst_middle = args.integer(8).?;
    for (0..outer) |group| {
        for (0..middle) |row| {
            const s: [*]const u8 = @ptrFromInt(src + group * src_outer + row * src_middle);
            const d: [*]u8 = @ptrFromInt(dst + group * dst_outer + row * dst_middle);
            @memcpy(d[0..inner], s[0..inner]);
        }
    }
}

const n = 3; // rows
const wf = [2]usize{ 20, 12 }; // fp32 columns a pair (pair 0 wider)
const wb = [2]usize{ 40, 24 }; // bf16 columns a pair
const pw = 8; // pairGather's fp32 width
const part_bytes = n * 64 * 4;

/// TP2 rank r's partial, element (row, col): distinct in every field, so a misplaced byte shows.
fn pf32(r: usize, row: usize, col: usize) u32 {
    return @intCast(0x4000_0000 | r << 20 | row << 10 | col);
}
fn pbf16(r: usize, row: usize, col: usize) u16 {
    return @intCast(r << 14 | row << 8 | col);
}
fn pgv(g: usize, row: usize, c: usize) u32 {
    return @intCast(0x8000_0000 | g << 20 | row << 10 | c);
}

fn addr(b: []align(16) u8) u64 {
    return @intFromPtr(b.ptr);
}

const Node = struct {
    g: usize,
    rank: link_fake.Rank,
    ops: *const ops_mod.Ops,
    bufs: [10][]align(16) u8, // sf, qf, cf, sb, qb, cb, sp, dp, parts, pad
    err: ?anyerror = null,

    fn run(nd: *Node) void {
        nd.work() catch |e| {
            nd.err = e;
        };
    }

    fn work(nd: *Node) !void {
        allow_strided = false;
        const cl = nd.rank.handle();
        var e: prompt.Engine = undefined;
        e.s = .{ .d = &driver, .handle = @ptrFromInt(0x20) };
        e.ops = nd.ops;
        e.comm = &cl;
        var t: prompt2d.Two = undefined;
        t.r = nd.g % 2;
        t.p = nd.g / 2;
        t.parts = addr(nd.bufs[8]);
        t.pad = addr(nd.bufs[9]);
        t.part_bytes = part_bytes;
        t.rings = null;
        const b = nd.bufs;
        try t.quarters(&e, addr(b[0]), addr(b[1]), n, wf, 4);
        try t.catRank(&e, addr(b[0]), addr(b[2]), n, wf, 4);
        try t.quarters(&e, addr(b[3]), addr(b[4]), n, wb, 2);
        try t.catRank(&e, addr(b[3]), addr(b[5]), n, wb, 2);
        try t.pairGather(&e, addr(b[6]), addr(b[7]), n, pw, 4);
    }
};

test "world 4: prompt2d quarters and catRank over link_fake (four ranks)" {
    const a = std.testing.allocator;
    const saved = gpu.launch.before;
    defer gpu.launch.before = saved;
    gpu.launch.before = hostCopy;
    driver.api.hipModuleLaunchKernel = accept;
    var o: ops_mod.Ops = undefined;
    o.copy16 = .{ .d = &driver, .handle = @ptrFromInt(copy16_id) };
    o.strided = .{ .d = &driver, .handle = @ptrFromInt(strided_id) };

    const wfa = wf[0] + wf[1];
    const wba = wb[0] + wb[1];
    const sizes = [10]usize{ n * 64 * 4, 2 * n * wfa * 4, n * wfa * 4, n * 64 * 2, 2 * n * wba * 2, n * wba * 2, n * pw * 4, 2 * n * pw * 4, 4 * part_bytes, part_bytes };
    var board: link_fake.Board = .init(4);
    var nodes: [4]Node = undefined;
    defer for (&nodes) |*nd| for (nd.bufs) |x| a.free(x);
    for (&nodes, 0..) |*nd, g| {
        nd.* = .{ .g = g, .rank = .{ .b = &board, .rank = @intCast(g) }, .ops = &o, .bufs = undefined };
        for (&nd.bufs, sizes) |*x, sz| {
            x.* = try a.alignedAlloc(u8, .@"16", sz);
            @memset(x.*, 0xa5);
        }
        // this node's columns of its rank's partials (pair 0's columns first), and its pairGather part
        const r = g % 2;
        const p = g / 2;
        const sf = std.mem.bytesAsSlice(u32, nd.bufs[0]);
        const sb = std.mem.bytesAsSlice(u16, nd.bufs[3]);
        const sp = std.mem.bytesAsSlice(u32, nd.bufs[6]);
        for (0..n) |row| {
            for (0..wf[p]) |c| sf[row * wf[p] + c] = pf32(r, row, (if (p == 0) 0 else wf[0]) + c);
            for (0..wb[p]) |c| sb[row * wb[p] + c] = pbf16(r, row, (if (p == 0) 0 else wb[0]) + c);
            for (0..pw) |c| sp[row * pw + c] = pgv(g, row, c);
        }
    }
    var th: [4]std.Thread = undefined;
    for (&th, &nodes) |*h, *nd| h.* = try std.Thread.spawn(.{}, Node.run, .{nd});
    for (th) |h| h.join();

    for (nodes, 0..) |nd, g| {
        if (nd.err) |e| return e;
        const r = g % 2;
        const p = g / 2;
        const qf = std.mem.bytesAsSlice(u32, nd.bufs[1]);
        const cf = std.mem.bytesAsSlice(u32, nd.bufs[2]);
        const qb = std.mem.bytesAsSlice(u16, nd.bufs[4]);
        const cb = std.mem.bytesAsSlice(u16, nd.bufs[5]);
        const dp = std.mem.bytesAsSlice(u32, nd.bufs[7]);
        for (0..n) |row| {
            for (0..2) |rr| {
                for (0..wfa) |c| try std.testing.expectEqual(pf32(rr, row, c), qf[(rr * n + row) * wfa + c]);
                for (0..wba) |c| try std.testing.expectEqual(pbf16(rr, row, c), qb[(rr * n + row) * wba + c]);
            }
            for (0..wfa) |c| try std.testing.expectEqual(pf32(r, row, c), cf[row * wfa + c]);
            for (0..wba) |c| try std.testing.expectEqual(pbf16(r, row, c), cb[row * wba + c]);
            for (0..2) |k| for (0..pw) |c| try std.testing.expectEqual(pgv(2 * p + k, row, c), dp[(k * n + row) * pw + c]);
        }
        try std.testing.expectEqual(@as(u64, 3), nd.rank.gathers); // quarters x2 and pairGather, no rings
        try std.testing.expectEqual(@as(u64, 2), nd.rank.exchanges); // catRank: one exchange with g ^ 2 each
    }
    std.debug.print("world 4: prompt2d quarters and catRank over link_fake (four ranks): fp32 and bf16 parts, pair widths {d}/{d}, pairGather: every node holds its TP2 rank's bytes\n", .{ wf[0], wf[1] });
}

const cap_n = 17;
const cap_w = 8;

const QNode = struct {
    g: usize,
    n: usize,
    w: [2]usize,
    esize: usize,
    swap12: bool,
    vgather: bool,
    rank: link_fake.Rank,
    ops: *const ops_mod.Ops,
    src: []align(16) u8,
    dst: []align(16) u8,
    parts: []align(16) u8,
    pad: []align(16) u8,
    copies: u32 = 0,
    strides: u32 = 0,
    err: ?anyerror = null,

    fn run(nd: *QNode) void {
        nd.work() catch |e| {
            nd.err = e;
        };
    }

    fn work(nd: *QNode) !void {
        allow_strided = nd.vgather;
        n_copy = 0;
        n_strided = 0;
        defer allow_strided = false;
        var cl = nd.rank.handle();
        try std.testing.expect(!cl.vgather);
        cl.swap12 = nd.swap12;
        cl.vgather = nd.vgather;
        var e: prompt.Engine = undefined;
        e.s = .{ .d = &driver, .handle = @ptrFromInt(0x20) };
        e.ops = nd.ops;
        e.comm = &cl;
        var t: prompt2d.Two = undefined;
        t.r = nd.g % 2;
        t.p = nd.g / 2;
        t.parts = addr(nd.parts);
        t.pad = addr(nd.pad);
        t.part_bytes = cap_n * cap_w * 4;
        t.rings = null;
        try t.quarters(&e, addr(nd.src), addr(nd.dst), nd.n, nd.w, nd.esize);
        nd.copies = n_copy;
        nd.strides = n_strided;
    }
};

fn fillSrc(nd: *QNode) void {
    @memset(nd.src, 0xa5);
    @memset(nd.dst, 0xa5);
    @memset(nd.parts, 0xa5);
    @memset(nd.pad, 0xa5);
    const r = nd.g % 2;
    const p = nd.g / 2;
    const cols = nd.w[p];
    const base: usize = if (p == 0) 0 else nd.w[0];
    if (nd.esize == 4) {
        const s = std.mem.bytesAsSlice(u32, nd.src);
        for (0..nd.n) |row| for (0..cols) |c| {
            s[row * cols + c] = pf32(r, row, base + c);
        };
    }
}

/// One quarters on four ranks. `dst` is rank 0's gathered rows (the other ranks match it on an equal-width gather).
fn oneGather(a: std.mem.Allocator, o: *const ops_mod.Ops, rows: usize, w: [2]usize, esize: usize, swap12: bool, vgather: bool) !struct {
    dst: []u8,
    copies: u32,
    strides: u32,
    gathers: u64,
    bytes: usize,
    site_swap12: bool,
} {
    const sizes = [4]usize{
        cap_n * cap_w * 4,
        2 * cap_n * (cap_w * 2) * 4,
        4 * cap_n * cap_w * 4,
        cap_n * cap_w * 4,
    };
    var board: link_fake.Board = .init(4);
    var nodes: [4]QNode = undefined;
    var storage: [4][4][]align(16) u8 = undefined;
    defer for (storage) |bufs| for (bufs) |x| a.free(x);
    for (&nodes, 0..) |*nd, g| {
        for (&storage[g], sizes) |*x, sz| x.* = try a.alignedAlloc(u8, .@"16", sz);
        nd.* = .{
            .g = g,
            .n = rows,
            .w = w,
            .esize = esize,
            .swap12 = swap12,
            .vgather = vgather,
            .rank = .{ .b = &board, .rank = @intCast(g) },
            .ops = o,
            .src = storage[g][0],
            .dst = storage[g][1],
            .parts = storage[g][2],
            .pad = storage[g][3],
        };
        fillSrc(nd);
    }
    var th: [4]std.Thread = undefined;
    for (&th, &nodes) |*h, *nd| h.* = try std.Thread.spawn(.{}, QNode.run, .{nd});
    for (th) |h| h.join();
    for (nodes) |nd| if (nd.err) |e| return e;
    const dst_bytes = 2 * rows * (w[0] + w[1]) * esize;
    const dst = try a.alloc(u8, dst_bytes);
    errdefer a.free(dst);
    @memcpy(dst, nodes[0].dst[0..dst_bytes]);
    for (nodes[1..]) |nd| try std.testing.expectEqualSlices(u8, dst, nd.dst[0..dst_bytes]);
    // a narrower pair pads before the gather, so its copy count is one higher; equal widths match on every rank
    if (w[0] == w[1]) for (nodes[1..]) |nd| {
        try std.testing.expectEqual(nodes[0].copies, nd.copies);
        try std.testing.expectEqual(nodes[0].strides, nd.strides);
    };
    const call = board.calls[0];
    try std.testing.expectEqual(coll.Phase.two_parts, call.site.phase);
    try std.testing.expectEqual(@as(u16, 0), call.site.layer);
    for (board.calls[1..4]) |c| {
        try std.testing.expectEqual(call.bytes, c.bytes);
        try std.testing.expectEqual(call.site.swap12, c.site.swap12);
    }
    return .{
        .dst = dst,
        .copies = nodes[0].copies,
        .strides = nodes[0].strides,
        .gathers = nodes[0].rank.gathers,
        .bytes = call.bytes,
        .site_swap12 = call.site.swap12,
    };
}

fn expectLayout(dst: []const u8, rows: usize, w: [2]usize) !void {
    const width = w[0] + w[1];
    const q = std.mem.bytesAsSlice(u32, dst);
    for (0..rows) |row| for (0..2) |rr| for (0..width) |c| {
        try std.testing.expectEqual(pf32(rr, row, c), q[(rr * rows + row) * width + c]);
    };
}

test "world 4: TF_DS_VGATHER quarters match the four copies, R = 1..8, and the default launches stay" {
    const a = std.testing.allocator;
    const saved = gpu.launch.before;
    defer gpu.launch.before = saved;
    gpu.launch.before = hostCopy;
    driver.api.hipModuleLaunchKernel = accept;
    var o: ops_mod.Ops = undefined;
    o.copy16 = .{ .d = &driver, .handle = @ptrFromInt(copy16_id) };
    o.strided = .{ .d = &driver, .handle = @ptrFromInt(strided_id) };
    const w = [2]usize{ cap_w, cap_w };

    // one row, swap12 on: both flags take the straight gather. link_fake writes rank order, not place order, so the
    // check is that the key does not change the call or the bytes, not the hardware permutation.
    const s_off = try oneGather(a, &o, 1, w, 4, true, false);
    defer a.free(s_off.dst);
    const s_on = try oneGather(a, &o, 1, w, 4, true, true);
    defer a.free(s_on.dst);
    try std.testing.expectEqualSlices(u8, s_off.dst, s_on.dst);
    try std.testing.expectEqual(@as(u32, 0), s_off.copies);
    try std.testing.expectEqual(@as(u32, 0), s_off.strides);
    try std.testing.expectEqual(@as(u32, 0), s_on.copies);
    try std.testing.expectEqual(@as(u32, 0), s_on.strides);
    try std.testing.expectEqual(@as(u64, 1), s_off.gathers);
    try std.testing.expectEqual(@as(u64, 1), s_on.gathers);
    try std.testing.expect(s_off.site_swap12 and s_on.site_swap12);
    try std.testing.expectEqual(w[0] * 4, s_off.bytes);
    try std.testing.expectEqual(s_off.bytes, s_on.bytes);

    for (1..9) |rows| {
        const off = try oneGather(a, &o, rows, w, 4, false, false);
        defer a.free(off.dst);
        const on = try oneGather(a, &o, rows, w, 4, false, true);
        defer a.free(on.dst);
        try expectLayout(off.dst, rows, w);
        try expectLayout(on.dst, rows, w);
        try std.testing.expectEqualSlices(u8, off.dst, on.dst);
        try std.testing.expectEqual(@as(u64, 1), off.gathers);
        try std.testing.expectEqual(@as(u64, 1), on.gathers);
        try std.testing.expect(!off.site_swap12 and !on.site_swap12);
        try std.testing.expectEqual(rows * w[0] * 4, off.bytes);
        try std.testing.expectEqual(off.bytes, on.bytes);
        // default off: four copy16, including R = 1 (no swap12). On: two copy16 for R = 2..8, never the byte kernel.
        try std.testing.expectEqual(@as(u32, 4), off.copies);
        try std.testing.expectEqual(@as(u32, 0), off.strides);
        if (rows == 1) {
            try std.testing.expectEqual(@as(u32, 4), on.copies);
            try std.testing.expectEqual(@as(u32, 0), on.strides);
        } else {
            try std.testing.expectEqual(@as(u32, 2), on.copies);
            try std.testing.expectEqual(@as(u32, 0), on.strides);
        }
    }

    // past 16 rows the two copy16 still run: the only cap is copy_rows16's grid.y (2*n <= 65535), not a decode window.
    const wide = try oneGather(a, &o, cap_n, w, 4, false, true);
    defer a.free(wide.dst);
    try expectLayout(wide.dst, cap_n, w);
    try std.testing.expectEqual(@as(u32, 2), wide.copies);
    try std.testing.expectEqual(@as(u32, 0), wide.strides);
    try std.testing.expectEqual(cap_n * w[0] * 4, wide.bytes);

    const uneven = [2]usize{ cap_w, cap_w / 2 };
    const un = try oneGather(a, &o, 3, uneven, 4, false, true);
    defer a.free(un.dst);
    try expectLayout(un.dst, 3, uneven);
    // pair 0 is the wider one: no pad, four copies. The hook counts one rank; rank 0 is pair 0.
    try std.testing.expectEqual(@as(u32, 4), un.copies);
    try std.testing.expectEqual(@as(u32, 0), un.strides);
    try std.testing.expect(!un.site_swap12);
    try std.testing.expectEqual(3 * cap_w * 4, un.bytes);

    // rb = 4 is not a multiple of 16, so copy_rows16 cannot run. The key falls back to the four copies
    // (the byte strided kernel), not two of them.
    const thin = [2]usize{ 1, 1 };
    const thin_on = try oneGather(a, &o, 3, thin, 4, false, true);
    defer a.free(thin_on.dst);
    try expectLayout(thin_on.dst, 3, thin);
    try std.testing.expectEqual(@as(u32, 0), thin_on.copies);
    try std.testing.expectEqual(@as(u32, 4), thin_on.strides);
    try std.testing.expect(!thin_on.site_swap12);
    try std.testing.expectEqual(3 * thin[0] * 4, thin_on.bytes);
}
