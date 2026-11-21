//! sf-disk for dsv41 on the host: settings, block rows, the identity, a pair's save and resume, a failed submit.
const std = @import("std");
const core = @import("core");
const disk = @import("disk.zig");
const pd = core.prompt_disk;
const Key = pd.Key;
const Settings = disk.Settings;
const Plane = disk.Plane;
const Parts = disk.Parts;
const block_tokens = disk.block_tokens;
const blockBytes = disk.blockBytes;
const rowsAt = disk.rowsAt;
const identityText = disk.identityText;
const seedText = disk.seedText;
const submitRuns = disk.submitRuns;
const warmTwin = disk.warmTwin;
const image_key = @import("lanes").image_key;
const fill_spans = @import("fill_spans.zig");
const lanes_mod = @import("lanes.zig");

test "settings: off unless a positive GiB; defaults; a value that is not a number turns the tier off" {
    const a = std.testing.allocator;
    const Env = struct {
        var vars: []const [2][]const u8 = &.{};
        fn get(name: [:0]const u8) ?[]const u8 {
            for (vars) |v| if (std.mem.eql(u8, v[0], name)) return v[1];
            return null;
        }
    };
    Env.vars = &.{.{ "HOME", "/h" }};
    try std.testing.expectEqual(@as(?Settings, null), try Settings.parse(a, Env.get));
    Env.vars = &.{ .{ "HOME", "/h" }, .{ "TF_DS_PROMPT_DISK_GIB", "0" } };
    try std.testing.expectEqual(@as(?Settings, null), try Settings.parse(a, Env.get));
    Env.vars = &.{ .{ "HOME", "/h" }, .{ "TF_DS_PROMPT_DISK_GIB", "256" } };
    const s = (try Settings.parse(a, Env.get)).?;
    defer a.free(s.dir);
    try std.testing.expectEqualStrings("/h/strixfold-cache/prefix", s.dir);
    try std.testing.expectEqual(@as(u64, 256 << 30), s.budget);
    try std.testing.expectEqual(@as(u64, 16 << 30), s.floor);
    try std.testing.expectEqual(@as(u64, 1 << 30), s.queue);
    try std.testing.expectEqual(@as(u32, 8192), s.min_tokens);
    Env.vars = &.{ .{ "TF_DS_PROMPT_DISK_GIB", "2.5" }, .{ "TF_DS_PROMPT_DISK_DIR", "/x" }, .{ "TF_DS_PROMPT_DISK_QUEUE_MIB", "64" }, .{ "TF_DS_PROMPT_DISK_FLOOR_GIB", "1" }, .{ "TF_DS_PROMPT_DISK_MIN_TOKENS", "100" } };
    const t = (try Settings.parse(a, Env.get)).?;
    defer a.free(t.dir);
    try std.testing.expectEqualStrings("/x", t.dir);
    try std.testing.expectEqual(@as(u64, 5 << 29), t.budget);
    try std.testing.expectEqual(@as(u64, 64 << 20), t.queue);
    try std.testing.expectEqual(@as(u64, 1 << 30), t.floor);
    try std.testing.expectEqual(@as(u32, block_tokens), t.min_tokens); // at least a block
    Env.vars = &.{.{ "TF_DS_PROMPT_DISK_GIB", "lots" }};
    try std.testing.expectError(error.BadSetting, Settings.parse(a, Env.get));
}

test "a block's rows: each plane's place in an extent and its bytes" {
    const p: Plane = .{ .base = 1 << 20, .row = 256, .ratio = 4 };
    try std.testing.expectEqual(@as(u64, block_tokens / 4 * 256), blockBytes(p));
    try std.testing.expectEqual(@as(u64, (1 << 20) + 4096 / 4 * 256), rowsAt(p, 4096, 1));
    try std.testing.expectEqual(@as(u64, (1 << 20) + (4096 + 2 * block_tokens) / 4 * 256), rowsAt(p, 4096, 3));
    // consecutive blocks' rows are consecutive in the plane (a run is one copy a plane)
    try std.testing.expectEqual(rowsAt(p, 0, 2), rowsAt(p, 0, 1) + blockBytes(p));
}

test "the identity: every input on its own line; the plan's conf= and disk= tokens left out" {
    const a = std.testing.allocator;
    const parts: Parts = .{ .rank = 1, .world = 2, .deploy = "70cfb8c", .bundle = "c178", .kit_sha256 = "abcd", .hip_runtime = 70051831, .arch = "gfx1151", .plan = "drafts=true pool=8192 conf=1234 kv=x disk=1", .engram = "e[t:1]", .token_map = "tm", .ds_kit = "dk", .window = 128, .snap_bytes = 6 << 20, .planes = "256:4" };
    const text = try identityText(a, parts);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "plan.pool=8192\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "conf") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "disk=") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "rank=1\n") != null);
    // a changed setting is a new namespace; the same inputs the same one
    var other = parts;
    other.plan = "drafts=true pool=16384 conf=1234 kv=x disk=1";
    const t2 = try identityText(a, other);
    defer a.free(t2);
    try std.testing.expect(!std.mem.eql(u8, &pd.namespaceOf(text), &pd.namespaceOf(t2)));
    var diff: [128]u8 = undefined;
    try std.testing.expectEqualStrings("plan.pool", pd.identityDiff(&diff, text, t2));
    var same = parts;
    same.plan = "drafts=true pool=8192 conf=9999 kv=x"; // serve.conf's hash and the disk flag do not change it
    const t3 = try identityText(a, same);
    defer a.free(t3);
    try std.testing.expectEqualStrings(text, t3);
}

test "a state resumed from memory keeps its disk twin recent: eviction takes the state used on neither tier" {
    const gpa = std.testing.allocator;
    var toks_a: [4 * block_tokens]u32 = undefined;
    for (&toks_a, 0..) |*x, i| x.* = @intCast(i * 31 % 129280);
    var toks_b: [4 * block_tokens]u32 = undefined;
    for (&toks_b, 0..) |*x, i| x.* = @intCast((i * 17 + 5) % 129280);
    var toks_c: [4 * block_tokens]u32 = undefined;
    for (&toks_c, 0..) |*x, i| x.* = @intCast((i * 13 + 9) % 129280);
    var ka: [4]Key = undefined;
    var kb: [4]Key = undefined;
    var kc: [4]Key = undefined;
    for ([_][]const u32{ &toks_a, &toks_b, &toks_c }, [_][]Key{ &ka, &kb, &kc }) |toks, keys| pd.chain(pd.seed("id"), std.mem.sliceAsBytes(toks), block_tokens * 4, keys);
    var rb: [128]u8 = undefined;
    const root = try testRoot(&rb, "warm");
    defer testRmTree(root);
    const cfg: pd.Config = .{ .root = root, .budget = 1 << 30, .floor = 0, .queue = 1 << 20, .rate = 0, .free_override = 1 << 40 };
    const t = try pd.Tier.open(gpa, cfg, "id", 0, test_slot, &test_planes);
    defer t.close();
    try testSave(t, &ka, 3, 0);
    try testSave(t, &kb, 3, 0);
    try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    t.touch(kb[2]); // b resumed from disk since: a is the least recently used
    // nothing to touch: no blocks, past the chain, a key not on disk
    try std.testing.expect(!warmTwin(t, &ka, 0));
    try std.testing.expect(!warmTwin(t, ka[0..2], 3));
    try std.testing.expect(!warmTwin(t, &kc, 3));
    try std.testing.expect(!warmTwin(t, &ka, 2)); // a's 2-block prefix is blocks only, no state
    // a resumed from its kept state in memory (fillStart): its twin is touched, so b is now the oldest
    try std.testing.expect(warmTwin(t, &ka, 3));
    // c needs exactly one state's room: the victim is b, the state used on neither tier since a's resume
    t.cfg.budget = t.bytes + 3 * (pd.head_bytes + pd.sum(&test_planes)) + t.stateFile(3) - 1;
    var p = try t.plan(kc[0..3], true, false);
    defer p.deinit(gpa);
    try std.testing.expect(p.result == .ok);
    try std.testing.expectEqual(@as(usize, 1), p.victims.len);
    try std.testing.expectEqualSlices(u8, &kb[2], &p.victims[0]);
    try std.testing.expectEqual(@as(?u32, 3), t.longest(&ka, 0, 4));
}

test "a block's key holds the whole digest of each picture that meets it, so key ids alike cannot resume it" {
    var ids: [3 * block_tokens]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i * 7 % 1000);
    const da = image_key.imageDigest("picture A");
    var db = da;
    db[20] ^= 1; // alike in the first 8 bytes; the ids below are A's for both (as if B's had collided)
    const h0 = pd.seed("id");
    var text: [3]Key = undefined;
    pd.chain(h0, std.mem.sliceAsBytes(&ids), block_tokens * 4, &text);
    var none: [3]Key = undefined;
    disk.chainKeys(h0, std.mem.sliceAsBytes(&ids), &.{}, &none);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&text), std.mem.sliceAsBytes(&none)); // text: pd.chain's
    var ka: [3]Key = undefined;
    var kb: [3]Key = undefined;
    disk.chainKeys(h0, std.mem.sliceAsBytes(&ids), &.{.{ .at = block_tokens - 2, .len = 4, .digest = da }}, &ka);
    disk.chainKeys(h0, std.mem.sliceAsBytes(&ids), &.{.{ .at = block_tokens - 2, .len = 4, .digest = db }}, &kb);
    for (ka, kb) |x, y| try std.testing.expect(!std.mem.eql(u8, &x, &y)); // the span meets blocks 1 and 2
    for (ka, text) |x, y| try std.testing.expect(!std.mem.eql(u8, &x, &y)); // nor is either the same ids' text key
    var late: [3]Key = undefined;
    disk.chainKeys(h0, std.mem.sliceAsBytes(&ids), &.{.{ .at = 2 * block_tokens + 5, .len = 4, .digest = da }}, &late);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(text[0..2]), std.mem.sliceAsBytes(late[0..2])); // before it: text
    try std.testing.expect(!std.mem.eql(u8, &text[2], &late[2]));
}

fn testRoot(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-sfdisk-dsv41-{s}-{d}", .{ tag, std.c.getpid() });
}

fn testRmTree(root: []const u8) void {
    var buf: [512]u8 = undefined;
    const r = std.fmt.bufPrintSentinel(&buf, "{s}", .{root}, 0) catch return;
    const d = std.c.opendir(r) orelse return;
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        var sub: [512]u8 = undefined;
        const p = std.fmt.bufPrint(&sub, "{s}/{s}", .{ root, name }) catch continue;
        var zb: [512]u8 = undefined;
        if (std.c.unlink(std.fmt.bufPrintSentinel(&zb, "{s}", .{p}, 0) catch continue) != 0) testRmTree(p);
    }
    _ = std.c.closedir(d);
    _ = std.c.rmdir(r);
}

const test_planes = [_]u64{ 64, 16 };
const test_slot = 96;

fn testParts(rank: u32) Parts {
    // the rank-local inputs may differ between the boxes (their own engram listing here): only the namespace sees them
    return .{ .rank = rank, .world = 2, .deploy = "4a5cb43", .bundle = "c178a012", .kit_sha256 = "73944f4e", .hip_runtime = 70051831, .arch = "gfx1151", .plan = "drafts=true pool=327680 conf=1 kv=x disk=1", .engram = if (rank == 0) "e[a:1]" else "e[a:1,b:2]", .token_map = "tm", .ds_kit = "dk", .window = 128, .snap_bytes = test_slot, .planes = "64:4,16:4" };
}

/// One rank's save of the state at k blocks, as diskSave and the disk_save handler make it (bytes depend on the rank).
fn testSave(t: *pd.Tier, chain_keys: []const Key, k: u32, rank: u8) !void {
    const gpa = std.testing.allocator;
    var p = try t.plan(chain_keys[0..k], true, false);
    defer p.deinit(gpa);
    try std.testing.expect(p.result == .ok);
    const Fill = struct {
        rank: u8,
        fn f(c: @This(), j0: u32, n: u32) ![]u8 {
            const data = try std.testing.allocator.alloc(u8, n * pd.sum(&test_planes));
            for (data, 0..) |*b, i| b.* = @truncate(c.rank *% 101 +% j0 *% 7 +% i);
            return data;
        }
    };
    try std.testing.expect(submitRuns(t, &p, chain_keys[0..k], 1, true, Fill{ .rank = rank }, Fill.f));
    const slot = try gpa.alloc(u8, test_slot);
    @memset(slot, rank +% 0x30);
    try t.submitState(chain_keys[0..k], slot, k);
}

test "rank 0 and rank 1 name a prompt's states alike: a pair's save, reconcile and resume all find them" {
    const gpa = std.testing.allocator;
    const p0 = testParts(0);
    const p1 = testParts(1);
    const id0 = try identityText(gpa, p0);
    defer gpa.free(id0);
    const id1 = try identityText(gpa, p1);
    defer gpa.free(id1);
    const s0 = try seedText(gpa, p0);
    defer gpa.free(s0);
    const s1 = try seedText(gpa, p1);
    defer gpa.free(s1);
    // each rank's files in a namespace of its own; one chain seed for both
    try std.testing.expect(!std.mem.eql(u8, &pd.namespaceOf(id0), &pd.namespaceOf(id1)));
    try std.testing.expectEqualStrings(s0, s1);
    try std.testing.expect(std.mem.indexOf(u8, s0, "rank") == null);
    // the same prompt on both ranks (rank 0 hashes its u32 tokens, rank 1 its i32 sequence: the same bytes)
    var toks: [6 * block_tokens]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(i * 31 % 129280);
    var seq: [6 * block_tokens]i32 = undefined;
    for (&seq, toks) |*q, x| q.* = @intCast(x);
    var k0: [6]Key = undefined;
    var k1: [6]Key = undefined;
    pd.chain(pd.seed(s0), std.mem.sliceAsBytes(&toks), block_tokens * 4, &k0);
    pd.chain(pd.seed(s1), std.mem.sliceAsBytes(&seq), block_tokens * 4, &k1);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&k0), std.mem.sliceAsBytes(&k1));
    // (the namespace's own text as the seed would name them differently: no hit would be possible)
    var w0: [1]Key = undefined;
    var w1: [1]Key = undefined;
    pd.chain(pd.seed(id0), std.mem.sliceAsBytes(&toks), block_tokens * 4, &w0);
    pd.chain(pd.seed(id1), std.mem.sliceAsBytes(&seq), block_tokens * 4, &w1);
    try std.testing.expect(!std.mem.eql(u8, &w0[0], &w1[0]));

    var rb0: [128]u8 = undefined;
    var rb1: [128]u8 = undefined;
    const root0 = try testRoot(&rb0, "pair0");
    const root1 = try testRoot(&rb1, "pair1");
    defer testRmTree(root0);
    defer testRmTree(root1);
    const cfg: pd.Config = .{ .root = undefined, .budget = 1 << 30, .floor = 0, .queue = 1 << 20, .rate = 0, .free_override = 1 << 40 };
    var c0 = cfg;
    c0.root = root0;
    var c1 = cfg;
    c1.root = root1;
    c1.follower = true;
    {
        const t0 = try pd.Tier.open(gpa, c0, id0, 0, test_slot, &test_planes);
        defer t0.close();
        const t1 = try pd.Tier.open(gpa, c1, id1, 1, test_slot, &test_planes);
        defer t1.close();
        // rank 0 keeps the states at 3 and 5 blocks; rank 1 hears each key and finds it in its own chain
        for ([_]u32{ 3, 5 }) |k| {
            try testSave(t0, &k0, k, 0);
            try std.testing.expectEqualSlices(u8, &k0[k - 1], &k1[k - 1]); // lanes.zig's disk_save check
            try testSave(t1, &k1, k, 1);
        }
        try std.testing.expect(t0.flush(5 * std.time.ns_per_s) and t1.flush(5 * std.time.ns_per_s));
    }
    // a restart: both reopen and reconcile, rank 0's list against rank 1's bitmap
    const t0 = try pd.Tier.open(gpa, c0, id0, 0, test_slot, &test_planes);
    defer t0.close();
    const t1 = try pd.Tier.open(gpa, c1, id1, 1, test_slot, &test_planes);
    defer t1.close();
    const listed = try t0.completeKeys();
    defer gpa.free(listed);
    try std.testing.expectEqual(@as(usize, 2), listed.len);
    const map = try pd.bitmap(gpa, t1, listed);
    defer gpa.free(map);
    for (0..listed.len) |i| try std.testing.expect(pd.bit(map, i)); // nothing to drop on either side
    // the resume: rank 0 picks the longest state of the prompt, both ranks read their own part back
    const k = t0.longest(&k0, 0, 6).?;
    try std.testing.expectEqual(@as(u32, 5), k);
    const keys = try gpa.alloc(Key, k);
    defer gpa.free(keys);
    const slot = try gpa.alloc(u8, test_slot);
    defer gpa.free(slot);
    const data = try gpa.alloc(u8, k * pd.sum(&test_planes));
    defer gpa.free(data);
    const file = try gpa.alloc(u8, pd.head_bytes + pd.sum(&test_planes));
    defer gpa.free(file);
    for ([_]*pd.Tier{ t0, t1 }, [_]u8{ 0, 1 }) |t, rank| {
        try std.testing.expectEqual(pd.Load.ok, t.readState(k0[k - 1], k, keys, slot));
        try std.testing.expect(t.readRun(keys, 1, data, file));
        try std.testing.expectEqual(rank +% 0x30, slot[0]);
    }
    try std.testing.expectEqual(pd.Outcome.hit, pd.Outcome.of(.ok, .ok));
}

test "a run whose submit fails queues nothing: its reservation comes back and its blocks go" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const gpa = failing.allocator();
    var rb: [128]u8 = undefined;
    const root = try testRoot(&rb, "submit");
    defer testRmTree(root);
    const t = try pd.Tier.open(gpa, .{ .root = root, .budget = 1 << 30, .floor = 0, .queue = 1 << 20, .rate = 0, .free_override = 1 << 40 }, "id", 0, test_slot, &test_planes);
    defer {
        failing.fail_index = std.math.maxInt(usize);
        t.close();
    }
    var toks: [3 * block_tokens]u32 = @splat(7);
    var keys: [3]Key = undefined;
    pd.chain(pd.seed("seed"), std.mem.sliceAsBytes(&toks), block_tokens * 4, &keys);
    var p = try t.plan(&keys, false, true); // a pass's streamed run of three blocks
    defer {
        failing.fail_index = std.math.maxInt(usize);
        p.deinit(gpa);
    }
    try std.testing.expect(t.queueRoom() < 1 << 20);
    const Fill = struct {
        f: *std.testing.FailingAllocator,
        fn f_(c: @This(), j0: u32, n: u32) ![]u8 {
            _ = j0;
            const data = try c.f.allocator().alloc(u8, n * pd.sum(&test_planes));
            c.f.fail_index = c.f.alloc_index; // the submit's copy of the run's keys fails
            return data;
        }
    };
    try std.testing.expect(!submitRuns(t, &p, &keys, 1, false, Fill{ .f = &failing }, Fill.f_));
    failing.fail_index = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(u64, 1 << 20), t.queueRoom());
    try std.testing.expectEqual(@as(usize, 0), t.blocks.count());
    try std.testing.expectEqual(@as(u64, 0), t.bytes);
}

/// A pair run: rank 0's and rank 1's free bytes on the state disk (df -B1 at the arm's start) and a
/// rank's block and slot bytes from its rank0.log open line.
const f_free: u64 = 4_481_019_904;
const d_free: u64 = 27_917_053_952;
const pair_block: u64 = 1_822_720;
const pair_slot: u64 = 6_684_672;

/// One rank's save of the state at k blocks at the pair's sizes (bytes depend on the rank and the block).
fn pairSave(t: *pd.Tier, chain_keys: []const Key, k: u32, rank: u8, planes: []const u64) !void {
    const gpa = std.testing.allocator;
    var p = try t.plan(chain_keys[0..k], true, false);
    defer p.deinit(gpa);
    try std.testing.expect(p.result == .ok);
    const Fill = struct {
        rank: u8,
        bb: u64,
        fn f(c: @This(), j0: u32, n: u32) ![]u8 {
            const data = try std.testing.allocator.alloc(u8, n * c.bb);
            for (data, 0..) |*b, i| b.* = @truncate(c.rank *% 101 +% j0 *% 7 +% i);
            return data;
        }
    };
    try std.testing.expect(submitRuns(t, &p, chain_keys[0..k], 1, true, Fill{ .rank = rank, .bb = pd.sum(planes) }, Fill.f));
    const slot = try gpa.alloc(u8, pair_slot);
    for (slot, 0..) |*b, i| b.* = @truncate(rank *% 13 +% i);
    try t.submitState(chain_keys[0..k], slot, k);
}

test "a long agent turn at the pair's sizes: pair 1's settings and rank 0's free bytes skip it at the floor; with room both ranks save it and read back the same bytes after a restart" {
    const gpa = std.testing.allocator;
    // pair 1's disk.conf, as serve_lib.sh hands it to the engine
    const Env = struct {
        fn get(name: [:0]const u8) ?[]const u8 {
            const vars = [_][2][]const u8{ .{ "TF_DS_PROMPT_DISK_GIB", "64" }, .{ "TF_DS_PROMPT_DISK_DIR", "/var/tmp/tfp-learned/pair1" }, .{ "TF_DS_PROMPT_DISK_FLOOR_GIB", "16" }, .{ "TF_DS_PROMPT_DISK_QUEUE_MIB", "1024" }, .{ "TF_DS_PROMPT_DISK_MIN_TOKENS", "8192" }, .{ "TF_DS_PROMPT_DISK_RATE_MIB", "1024" } };
            for (vars) |v| if (std.mem.eql(u8, v[0], name)) return v[1];
            return null;
        }
    };
    const s = (try Settings.parse(gpa, Env.get)).?;
    defer gpa.free(s.dir);
    try std.testing.expectEqual(@as(u64, 16 << 30), s.floor);
    // a long agent turn: the arm's median prompt (35,267 tokens) keeps its state at the last block end before len - 128
    const len: usize = 35_267;
    const k: u32 = @intCast((len - 128) / block_tokens);
    try std.testing.expectEqual(@as(u32, 17), k);
    try std.testing.expect(k * block_tokens >= s.min_tokens);
    var toks: [18 * block_tokens]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast((i * 2654435761) % 129280);
    const p0 = testParts(0);
    const p1 = testParts(1);
    const id0 = try identityText(gpa, p0);
    defer gpa.free(id0);
    const id1 = try identityText(gpa, p1);
    defer gpa.free(id1);
    const seed = try seedText(gpa, p0);
    defer gpa.free(seed);
    var keys: [17]Key = undefined;
    pd.chain(pd.seed(seed), std.mem.sliceAsBytes(toks[0 .. 17 * block_tokens]), block_tokens * 4, &keys);
    const planes = [_]u64{ pair_block - 4096, 4096 }; // two planes summing to the pair's block bytes
    var rb0: [128]u8 = undefined;
    var rb1: [128]u8 = undefined;
    const root0 = try testRoot(&rb0, "turn0");
    const root1 = try testRoot(&rb1, "turn1");
    defer testRmTree(root0);
    defer testRmTree(root1);
    const base: pd.Config = .{ .root = undefined, .budget = s.budget, .floor = s.floor, .queue = s.queue, .rate = s.rate, .block_tokens = block_tokens };
    var c0 = base;
    c0.root = root0;
    var c1 = base;
    c1.root = root1;
    c1.follower = true;
    // pair 1 as run: rank 0 with 4.48 GB free under a 16 GiB floor: the tier opens, is not writable, and skips
    {
        c0.free_override = f_free;
        const t0 = try pd.Tier.open(gpa, c0, id0, 0, pair_slot, &planes);
        defer t0.close();
        try std.testing.expect(!t0.writable() and t0.floor_warned);
        var p = try t0.plan(&keys, true, false);
        defer p.deinit(gpa);
        try std.testing.expectEqual(pd.Reason.floor, p.result.skip);
        try std.testing.expectEqual(@as(u64, 1), t0.counts.floor);
        try std.testing.expectEqual(@as(u64, 0), t0.bytes);
    }
    // the rerun arm (kit diskr.conf): floor 1 GiB, budget 2.5 GiB on the same 4.48 GB: writable, the save fires
    c0.floor = 1 << 30;
    c0.budget = 5 << 29;
    c1.floor = 1 << 30;
    c1.budget = 5 << 29;
    c1.free_override = d_free;
    {
        const t0 = try pd.Tier.open(gpa, c0, id0, 0, pair_slot, &planes);
        defer t0.close();
        const t1 = try pd.Tier.open(gpa, c1, id1, 1, pair_slot, &planes);
        defer t1.close();
        try std.testing.expect(t0.writable() and !t0.floor_warned and t1.writable());
        try pairSave(t0, &keys, k, 0, &planes);
        try pairSave(t1, &keys, k, 1, &planes);
        try std.testing.expect(t0.flush(30 * std.time.ns_per_s) and t1.flush(30 * std.time.ns_per_s));
        try std.testing.expectEqual(@as(u64, 1), t0.counts.saved);
        try std.testing.expectEqual(@as(u64, 1), t1.counts.saved);
        try std.testing.expectEqual(@as(u64, 0), t0.counts.floor);
        // one rank's files for this turn: 17 blocks and the state (about 37.7 MB), well inside the 2.5 GiB budget
        try std.testing.expectEqual(17 * (pd.head_bytes + pair_block) + t0.stateFile(17), t0.bytes);
    }
    // the restart: both reopen; rank 0 finds the state for the next turn's prompt and both read back what they wrote
    const t0 = try pd.Tier.open(gpa, c0, id0, 0, pair_slot, &planes);
    defer t0.close();
    const t1 = try pd.Tier.open(gpa, c1, id1, 1, pair_slot, &planes);
    defer t1.close();
    const listed = try t0.completeKeys();
    defer gpa.free(listed);
    try std.testing.expectEqual(@as(usize, 1), listed.len);
    const map = try pd.bitmap(gpa, t1, listed);
    defer gpa.free(map);
    try std.testing.expect(pd.bit(map, 0));
    var next: [18]Key = undefined; // the next turn's prompt: this one and one more block
    pd.chain(pd.seed(seed), std.mem.sliceAsBytes(&toks), block_tokens * 4, &next);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&keys), std.mem.sliceAsBytes(next[0..17]));
    try std.testing.expectEqual(@as(?u32, 17), t0.longest(&next, 0, 18));
    const got = try gpa.alloc(Key, k);
    defer gpa.free(got);
    const slot = try gpa.alloc(u8, pair_slot);
    defer gpa.free(slot);
    const data = try gpa.alloc(u8, k * pair_block);
    defer gpa.free(data);
    const file = try gpa.alloc(u8, pd.head_bytes + pair_block);
    defer gpa.free(file);
    const want = try gpa.alloc(u8, k * pair_block);
    defer gpa.free(want);
    for ([_]*pd.Tier{ t0, t1 }, [_]u8{ 0, 1 }) |t, rank| {
        try std.testing.expectEqual(pd.Load.ok, t.readState(keys[k - 1], k, got, slot));
        for (slot, 0..) |b, i| if (b != @as(u8, @truncate(rank *% 13 +% i))) return error.TestSlotDiffers;
        try std.testing.expect(t.readRun(got, 1, data, file));
        // the run as written (one fill from block 1, plane-major as submitRuns copied it)
        for (want, 0..) |*b, i| b.* = @truncate(rank *% 101 +% @as(u32, 1) *% 7 +% i);
        try std.testing.expectEqualSlices(u8, want, data);
    }
    try std.testing.expectEqual(pd.Outcome.hit, pd.Outcome.of(.ok, .ok));
}

/// Disk tier eviction: a disk that fills as the tier writes. Its free bytes are `start` less the bytes of
/// the regular files under the tier's root (the dir's parent), so an eviction's room shows only once its files are unlinked.
const FillingDisk = struct {
    var start: u64 = 0;
    fn free(dir: [:0]const u8) u64 {
        const cut = std.mem.lastIndexOfScalar(u8, dir, '/') orelse return 0;
        return start -| duTree(dir[0..cut]);
    }
    fn duTree(path: []const u8) u64 {
        var zb: [600]u8 = undefined;
        const z = std.fmt.bufPrintSentinel(&zb, "{s}", .{path}, 0) catch return 0;
        const d = std.c.opendir(z) orelse {
            const fd = std.c.open(z, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
            if (fd < 0) return 0;
            defer _ = std.c.close(fd);
            var sx: std.os.linux.Statx = undefined;
            if (std.os.linux.errno(std.os.linux.statx(fd, "", 0x1000, .{ .SIZE = true }, &sx)) != .SUCCESS) return 0;
            return sx.size;
        };
        defer _ = std.c.closedir(d);
        var total: u64 = 0;
        while (std.c.readdir(d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
            var sub: [600]u8 = undefined;
            total += duTree(std.fmt.bufPrint(&sub, "{s}/{s}", .{ path, name }) catch continue);
        }
        return total;
    }
};

/// Prompt `tag`'s run and slot bytes on `rank` (one run from block 1, plane-major, as submitRuns copies it).
fn evictRunByte(rank: u8, tag: u8, i: usize) u8 {
    return @truncate(@as(usize, rank) *% 101 +% @as(usize, tag) *% 31 +% 7 +% i);
}
fn evictSlotByte(rank: u8, tag: u8, i: usize) u8 {
    return @truncate(@as(usize, rank) *% 13 +% @as(usize, tag) *% 17 +% i);
}

/// One rank's submit of a plan that stands: prompt `tag`'s k blocks and its state.
fn evictSubmit(t: *pd.Tier, p: *const pd.Plan, keys: []const Key, rank: u8, tag: u8) !void {
    try std.testing.expect(p.result == .ok);
    const Fill = struct {
        rank: u8,
        tag: u8,
        bb: u64,
        fn f(c: @This(), j0: u32, n: u32) ![]u8 {
            try std.testing.expectEqual(@as(u32, 1), j0);
            const data = try std.testing.allocator.alloc(u8, n * c.bb);
            for (data, 0..) |*b, i| b.* = evictRunByte(c.rank, c.tag, i);
            return data;
        }
    };
    try std.testing.expect(submitRuns(t, p, keys, 1, true, Fill{ .rank = rank, .tag = tag, .bb = pd.sum(t.planes) }, Fill.f));
    const slot = try std.testing.allocator.alloc(u8, pair_slot);
    for (slot, 0..) |*b, i| b.* = evictSlotByte(rank, tag, i);
    try t.submitState(keys, slot, @intCast(keys.len));
}

/// A pair's save of prompt `tag` as lanes makes it: rank 0 plans (and may evict), rank 1 drops rank 0's victims
/// (the disk_drop frame) and then plans its own; both write. Returns rank 0's victims (gpa) after both have flushed.
fn evictTurn(t0: *pd.Tier, t1: *pd.Tier, keys: []const Key, tag: u8) ![]Key {
    const gpa = std.testing.allocator;
    var p0 = try t0.plan(keys, true, false);
    defer p0.deinit(gpa);
    t1.drop(p0.victims);
    try evictSubmit(t0, &p0, keys, 0, tag);
    var p1 = try t1.plan(keys, true, false);
    defer p1.deinit(gpa);
    try evictSubmit(t1, &p1, keys, 1, tag);
    try std.testing.expect(t0.flush(30 * std.time.ns_per_s) and t1.flush(30 * std.time.ns_per_s));
    return gpa.dupe(Key, p0.victims);
}

/// Both ranks read state `keys` back and find prompt `tag`'s bytes, slot and every block.
fn evictReadBack(t: *pd.Tier, keys: []const Key, rank: u8, tag: u8) !void {
    const gpa = std.testing.allocator;
    const k: u32 = @intCast(keys.len);
    const got = try gpa.alloc(Key, k);
    defer gpa.free(got);
    const slot = try gpa.alloc(u8, pair_slot);
    defer gpa.free(slot);
    const data = try gpa.alloc(u8, k * pair_block);
    defer gpa.free(data);
    const file = try gpa.alloc(u8, pd.head_bytes + pair_block);
    defer gpa.free(file);
    try std.testing.expectEqual(pd.Load.ok, t.readState(keys[k - 1], k, got, slot));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(keys), std.mem.sliceAsBytes(got));
    for (slot, 0..) |b, i| if (b != evictSlotByte(rank, tag, i)) return error.TestSlotDiffers;
    try std.testing.expect(t.readRun(got, 1, data, file));
    for (data, 0..) |b, i| if (b != evictRunByte(rank, tag, i)) return error.TestRunDiffers;
}

test "the floor at the pair's sizes: a disk filled to the floor evicts rank 0's least recently used state for the next save and rank 1 mirrors it; the floor holds, nothing outside the tier's dir goes, an evicted state re-reads from a fresh save, a kept one reads back identical after a restart, and a disk another tenant filled evicts nothing" {
    const gpa = std.testing.allocator;
    const k: u32 = 17; // a long agent turn (the median 35,267-token prompt), as in the pair-1 test above
    const p0 = testParts(0);
    const p1 = testParts(1);
    const id0 = try identityText(gpa, p0);
    defer gpa.free(id0);
    const id1 = try identityText(gpa, p1);
    defer gpa.free(id1);
    const seed = try seedText(gpa, p0);
    defer gpa.free(seed);
    // five prompts that share no block: A B C D fill and churn the disk, E comes when another tenant filled it
    var keys: [5][17]Key = undefined;
    var toks: [17 * block_tokens]u32 = undefined;
    for (&keys, 0..) |*ks, tag| {
        for (&toks, 0..) |*x, i| x.* = @intCast((i * 2654435761 + tag * 7919 + 1) % 129280);
        pd.chain(pd.seed(seed), std.mem.sliceAsBytes(&toks), block_tokens * 4, ks);
    }
    const A = 0;
    const B = 1;
    const C = 2;
    const D = 3;
    const E = 4;
    const planes = [_]u64{ pair_block - 4096, 4096 };
    var rb0: [128]u8 = undefined;
    var rb1: [128]u8 = undefined;
    const root0 = try testRoot(&rb0, "evict0");
    const root1 = try testRoot(&rb1, "evict1");
    testRmTree(root0);
    testRmTree(root1);
    defer testRmTree(root0);
    defer testRmTree(root1);
    // the rerun arm's settings (kit diskr.conf): floor 1 GiB, budget 2.5 GiB (it never binds here), queue and rate 1 GiB
    const floor: u64 = 1 << 30;
    const base: pd.Config = .{ .root = undefined, .budget = 5 << 29, .floor = floor, .queue = 1 << 30, .rate = 1 << 30, .block_tokens = block_tokens, .free_fn = FillingDisk.free };
    var c0 = base;
    c0.root = root0;
    var c1 = base;
    c1.root = root1;
    c1.follower = true;
    // one save's files on a rank (17 blocks and the state, about 37.7 MB); the disk holds the floor and 3.5 of them
    const save_bytes = k * (pd.head_bytes + pair_block) + (pd.head_bytes + 32 * k + pair_slot);
    FillingDisk.start = floor + 3 * save_bytes + save_bytes / 2;
    // a file of another owner beside the tier's dir, in its root: counted as used, never touched
    var fb: [200]u8 = undefined;
    {
        const t0 = try pd.Tier.open(gpa, c0, id0, 0, pair_slot, &planes);
        defer t0.close();
        const t1 = try pd.Tier.open(gpa, c1, id1, 1, pair_slot, &planes);
        defer t1.close();
        try std.testing.expectEqual(save_bytes, t0.stateFile(k) + k * (pd.head_bytes + pair_block));
        const foreign = try std.fmt.bufPrintSentinel(&fb, "{s}/not-sf-disk.bin", .{root0}, 0);
        {
            const fd = std.c.open(foreign, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
            try std.testing.expect(fd >= 0);
            const bytes4k: [4096]u8 = @splat(0x5a);
            try std.testing.expectEqual(@as(isize, 4096), std.c.write(fd, &bytes4k, bytes4k.len));
            _ = std.c.close(fd);
        }
        // A, B and C fill the disk to the floor: no eviction yet
        for ([_]usize{ A, B, C }) |tag| {
            const v = try evictTurn(t0, t1, &keys[tag], @intCast(tag));
            defer gpa.free(v);
            try std.testing.expectEqual(@as(usize, 0), v.len);
        }
        try std.testing.expectEqual(@as(u64, 3), t0.counts.saved);
        try std.testing.expectEqual(@as(u64, 3), t1.counts.saved);
        for ([_]*pd.Tier{ t0, t1 }) |t| try std.testing.expect(t.room() >= floor and t.room() < floor + save_bytes); // the next save does not fit
        try std.testing.expect(t0.writable()); // (rank 0 can make the room from its own states)
        // A was resumed since (finish .hit touches it): B is now the least recently used
        t0.touch(keys[A][k - 1]);
        // D: rank 0 evicts B for the floor, rank 1 drops B too, both save D, and the floor holds on both disks
        {
            const v = try evictTurn(t0, t1, &keys[D], D);
            defer gpa.free(v);
            try std.testing.expectEqual(@as(usize, 1), v.len);
            try std.testing.expectEqualSlices(u8, &keys[B][k - 1], &v[0]);
        }
        try std.testing.expectEqual(@as(u64, 1), t0.counts.evicted);
        try std.testing.expectEqual(@as(u64, 1), t0.counts.evicted_floor);
        try std.testing.expectEqual(@as(u64, 0), t0.counts.floor);
        try std.testing.expectEqual(@as(u64, 4), t0.counts.saved);
        try std.testing.expectEqual(@as(u64, 4), t1.counts.saved);
        try std.testing.expectEqual(@as(u64, 0), t1.counts.skipped);
        for ([_]*pd.Tier{ t0, t1 }) |t| {
            try std.testing.expect(t.freeBytes() >= floor);
            try std.testing.expectEqual(@as(u64, 0), t.unlinking);
            try std.testing.expectEqual(@as(u32, 3), t.states.count());
            try std.testing.expectEqual(3 * save_bytes, t.bytes);
        }
        // the evicted state: no rank lists it, its files are gone (a resume finds no state and the pass runs from 0)
        for ([_]*pd.Tier{ t0, t1 }) |t| {
            try std.testing.expect(!t.hasComplete(keys[B][k - 1]));
            try std.testing.expectEqual(@as(?u32, null), t.longest(&keys[B], 0, k));
            var pb: [1400]u8 = undefined;
            try std.testing.expect(std.c.access(try t.statePath(&pb, keys[B][k - 1], false), 0) != 0);
            for (keys[B]) |bk| try std.testing.expect(std.c.access(try t.blockPath(&pb, bk, false), 0) != 0);
        }
        try std.testing.expectEqual(@as(u64, 4096), FillingDisk.duTree(foreign));
        // B's next turn ran from 0 and saves again: the least recently used is now C (A was touched, D is newer)
        {
            const v = try evictTurn(t0, t1, &keys[B], B);
            defer gpa.free(v);
            try std.testing.expectEqual(@as(usize, 1), v.len);
            try std.testing.expectEqualSlices(u8, &keys[C][k - 1], &v[0]);
        }
        try std.testing.expectEqual(@as(u64, 2), t0.counts.evicted_floor);
        for ([_]*pd.Tier{ t0, t1 }) |t| try std.testing.expect(t.freeBytes() >= floor);
    }
    // the restart: both reopen; A (kept) and B (evicted, then saved again) read back identical on both ranks; C is gone
    const t0 = try pd.Tier.open(gpa, c0, id0, 0, pair_slot, &planes);
    defer t0.close();
    const t1 = try pd.Tier.open(gpa, c1, id1, 1, pair_slot, &planes);
    defer t1.close();
    const listed = try t0.completeKeys();
    defer gpa.free(listed);
    try std.testing.expectEqual(@as(usize, 3), listed.len);
    for ([_]*pd.Tier{ t0, t1 }, [_]u8{ 0, 1 }) |t, rank| {
        try std.testing.expectEqual(@as(?u32, k), t.longest(&keys[A], 0, k));
        try std.testing.expectEqual(@as(?u32, k), t.longest(&keys[B], 0, k));
        try std.testing.expectEqual(@as(?u32, null), t.longest(&keys[C], 0, k));
        try evictReadBack(t, &keys[A], rank, A);
        try evictReadBack(t, &keys[B], rank, B);
        try evictReadBack(t, &keys[D], rank, D);
    }
    // another tenant fills rank 0's disk past what all its own states could free: E skips, nothing is evicted
    {
        const big = try std.fmt.bufPrintSentinel(&fb, "{s}/other-tenant.bin", .{root0}, 0);
        const fd = std.c.open(big, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        try std.testing.expect(fd >= 0);
        try std.testing.expectEqual(@as(c_int, 0), std.c.ftruncate(fd, @intCast(FillingDisk.start))); // sparse: its size counts
        _ = std.c.close(fd);
    }
    var pe = try t0.plan(&keys[E], true, false);
    defer pe.deinit(gpa);
    try std.testing.expectEqual(pd.Reason.floor, pe.result.skip);
    try std.testing.expectEqual(@as(usize, 0), pe.victims.len);
    try std.testing.expectEqual(@as(u64, 0), t0.counts.evicted);
    try std.testing.expectEqual(@as(u32, 3), t0.states.count());
    try std.testing.expect(!t0.writable());
}

/// Node g of the four-node split (TP2 rank g % 2 of pair g / 2): the pair's parts with world=4 and its own rank.
fn quadParts(g: u32) Parts {
    var p = testParts(g % 2);
    p.rank = g;
    p.world = 4;
    return p;
}

const quad_tags = [4][]const u8{ "quad0", "quad1", "quad2", "quad3" };

test "world 4: four nodes each write and read their own namespace; rank 0's reconcile and evictions hold on all three followers" {
    const gpa = std.testing.allocator;
    var ids: [4][]u8 = undefined;
    var seeds: [4][]u8 = undefined;
    for (0..4) |g| {
        ids[g] = try identityText(gpa, quadParts(@intCast(g)));
        seeds[g] = try seedText(gpa, quadParts(@intCast(g)));
    }
    defer for (ids, seeds) |i, s| {
        gpa.free(i);
        gpa.free(s);
    };
    // four namespaces, none of them a pair rank's; one chain seed for all four, not the pair's
    for (0..4) |a| for (a + 1..4) |b| try std.testing.expect(!std.mem.eql(u8, &pd.namespaceOf(ids[a]), &pd.namespaceOf(ids[b])));
    for (0..2) |r| {
        const pid = try identityText(gpa, testParts(@intCast(r)));
        defer gpa.free(pid);
        for (0..4) |g| try std.testing.expect(!std.mem.eql(u8, &pd.namespaceOf(pid), &pd.namespaceOf(ids[g])));
    }
    for (seeds[1..]) |s| try std.testing.expectEqualStrings(seeds[0], s);
    const pseed = try seedText(gpa, testParts(0));
    defer gpa.free(pseed);
    try std.testing.expect(!std.mem.eql(u8, pseed, seeds[0]));
    try std.testing.expect(std.mem.indexOf(u8, ids[3], "rank=3\n") != null and std.mem.indexOf(u8, ids[3], "world=4\n") != null);
    // one prompt, the same keys on every node
    var toks: [6 * block_tokens]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(i * 37 % 129280);
    var keys: [6]Key = undefined;
    pd.chain(pd.seed(seeds[0]), std.mem.sliceAsBytes(&toks), block_tokens * 4, &keys);

    var rbs: [4][128]u8 = undefined;
    var cfgs: [4]pd.Config = undefined;
    for (0..4) |g| {
        const root = try testRoot(&rbs[g], quad_tags[g]);
        testRmTree(root);
        cfgs[g] = .{ .root = root, .budget = 1 << 30, .floor = 0, .queue = 1 << 20, .rate = 0, .free_override = 1 << 40, .follower = g != 0 };
    }
    defer for (cfgs) |c| testRmTree(c.root);
    {
        var ts: [4]*pd.Tier = undefined;
        for (0..4) |g| ts[g] = try pd.Tier.open(gpa, cfgs[g], ids[g], @intCast(g), test_slot, &test_planes);
        defer for (ts) |t| t.close();
        // rank 0 keeps the states at 3 and 5 blocks and every follower writes its own part; node 3 skipped the first
        for ([_]u32{ 3, 5 }) |k| for (ts, 0..) |t, g| {
            if (g == 3 and k == 3) continue;
            try testSave(t, &keys, k, @intCast(g));
        };
        for (ts) |t| try std.testing.expect(t.flush(5 * std.time.ns_per_s));
        // each node's files under its own namespace directory: its state file there, in no other node's
        var pb: [1400]u8 = undefined;
        for (ts, 0..) |t, g| {
            const hex = std.fmt.bytesToHex(pd.namespaceOf(ids[g]), .lower);
            try std.testing.expectEqualStrings(&hex, &t.ns_hex);
            try std.testing.expect(std.mem.endsWith(u8, t.dir, &hex));
            try std.testing.expect(std.c.access(try t.statePath(&pb, keys[4], false), 0) == 0);
        }
        try std.testing.expect(std.c.access(try ts[3].statePath(&pb, keys[2], false), 0) != 0);
    }
    // a restart: every node reopens; rank 0 lists its states, each follower answers its bitmap (ANDed as
    // lanes.syncAnswers does), and the state node 3 lacks goes on every node (rank 0 tells the others: disk_drop)
    var ts: [4]*pd.Tier = undefined;
    for (0..4) |g| ts[g] = try pd.Tier.open(gpa, cfgs[g], ids[g], @intCast(g), test_slot, &test_planes);
    defer for (ts) |t| t.close();
    const listed = try ts[0].completeKeys();
    defer gpa.free(listed);
    try std.testing.expectEqual(@as(usize, 2), listed.len);
    const have = try gpa.alloc(u8, (listed.len + 7) / 8);
    defer gpa.free(have);
    @memset(have, 0xff);
    for (ts[1..]) |t| {
        const map = try pd.bitmap(gpa, t, listed);
        defer gpa.free(map);
        for (have, map) |*h, b| h.* &= b;
    }
    const gone = try disk.notOnEvery(gpa, listed, have);
    defer gpa.free(gone);
    try std.testing.expectEqual(@as(usize, 1), gone.len);
    try std.testing.expectEqualSlices(u8, &keys[2], &gone[0]);
    for (ts) |t| t.drop(gone);
    for (ts) |t| {
        try std.testing.expectEqual(@as(u32, 1), t.states.count());
        try std.testing.expectEqual(@as(?u32, 5), t.longest(&keys, 0, 6));
    }
    // the resume: every node reads its own part back (slot and blocks carry its rank), and the outcome is a hit
    const k: u32 = 5;
    const got = try gpa.alloc(Key, k);
    defer gpa.free(got);
    const slot = try gpa.alloc(u8, test_slot);
    defer gpa.free(slot);
    const data = try gpa.alloc(u8, k * pd.sum(&test_planes));
    defer gpa.free(data);
    const file = try gpa.alloc(u8, pd.head_bytes + pd.sum(&test_planes));
    defer gpa.free(file);
    var loads: [4]pd.Load = undefined;
    for (ts, 0..) |t, g| {
        loads[g] = t.readState(keys[k - 1], k, got, slot);
        try std.testing.expectEqual(pd.Load.ok, loads[g]);
        try std.testing.expect(t.readRun(got, 1, data, file));
        try std.testing.expectEqual(@as(u8, @intCast(g)) +% 0x30, slot[0]);
        // run bytes are plane-major (rangeOffset): plane 0's first block starts the run
        try std.testing.expectEqual(@as(u8, @truncate(@as(u32, @intCast(g)) *% 101 +% 7)), data[0]);
    }
    try std.testing.expectEqual(pd.Outcome.hit, pd.Outcome.ofAll(loads[0], loads[1..]));
    // evictions: under a budget of one and a quarter saves rank 0 evicts the least recently used state for the next
    // prompt; each follower drops rank 0's victims (disk_drop) before it plans its own part, and all four agree
    const save5 = 5 * (pd.head_bytes + pd.sum(&test_planes)) + ts[0].stateFile(5);
    var toks2: [5 * block_tokens]u32 = undefined;
    for (&toks2, 0..) |*x, i| x.* = @intCast((i * 2654435761 + 11) % 129280);
    var keys2: [5]Key = undefined;
    pd.chain(pd.seed(seeds[0]), std.mem.sliceAsBytes(&toks2), block_tokens * 4, &keys2);
    for (ts) |t| t.cfg.budget = save5 + save5 / 4;
    var p0 = try ts[0].plan(&keys2, true, false);
    defer p0.deinit(gpa);
    try std.testing.expect(p0.result == .ok);
    try std.testing.expectEqual(@as(usize, 1), p0.victims.len);
    try std.testing.expectEqualSlices(u8, &keys[4], &p0.victims[0]);
    for (ts[1..]) |t| t.drop(p0.victims);
    try testSaveWith(ts[0], &p0, &keys2, 0);
    for (ts[1..], 1..) |t, g| {
        var p = try t.plan(&keys2, true, false);
        defer p.deinit(gpa);
        try std.testing.expect(p.result == .ok); // (a follower that kept the victim would refuse on its budget)
        try testSaveWith(t, &p, &keys2, @intCast(g));
    }
    for (ts) |t| try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    for (ts) |t| {
        try std.testing.expectEqual(@as(u32, 1), t.states.count());
        try std.testing.expect(!t.hasComplete(keys[4]) and t.hasComplete(keys2[4]));
        try std.testing.expect(t.bytes <= t.cfg.budget);
    }
    std.debug.print("world 4: sf-disk on four nodes: 4 namespaces (rank=g world=4), one chain seed; reconcile dropped {d} of {d} states (node 3 lacked it) on every node; resume {s} with each node's own bytes; rank 0 evicted {d} state for the budget and 3 followers mirrored it\n", .{ gone.len, listed.len, @tagName(pd.Outcome.ofAll(loads[0], loads[1..])), p0.victims.len });
}

/// One rank's submit of a plan that stands (testSave's bytes for a plan already made).
fn testSaveWith(t: *pd.Tier, p: *const pd.Plan, chain_keys: []const Key, rank: u8) !void {
    const k: u32 = @intCast(chain_keys.len);
    const Fill = struct {
        rank: u8,
        fn f(c: @This(), j0: u32, n: u32) ![]u8 {
            const data = try std.testing.allocator.alloc(u8, n * pd.sum(&test_planes));
            for (data, 0..) |*b, i| b.* = @truncate(c.rank *% 101 +% j0 *% 7 +% i);
            return data;
        }
    };
    try std.testing.expect(submitRuns(t, p, chain_keys, 1, true, Fill{ .rank = rank }, Fill.f));
    const slot = try std.testing.allocator.alloc(u8, test_slot);
    @memset(slot, rank +% 0x30);
    try t.submitState(chain_keys, slot, k);
}

test "rank 1 keys an image prompt's blocks as rank 0 does, from fill_begin's spans, a span across a block end too" {
    const gpa = std.testing.allocator;
    const image: u32 = 128815; // the image token: rank 0's prompt holds it, its sequence -1 there
    var prompt: [3 * block_tokens]u32 = undefined;
    for (&prompt, 0..) |*x, i| x.* = @intCast(i * 31 % 129280 % image);
    const spans = [_]image_key.KeySpan{ .{ .at = block_tokens - 3, .len = 10, .digest = image_key.imageDigest("picture A") }, .{ .at = 2 * block_tokens + 100, .len = 64, .digest = image_key.imageDigest("picture B") } };
    var seq0: [prompt.len]i32 = undefined;
    for (&seq0, prompt) |*q, x| q.* = @intCast(x);
    for (spans) |sp| for (prompt[sp.at..][0..sp.len], seq0[sp.at..][0..sp.len]) |*x, *q| {
        x.* = image;
        q.* = -1;
    };
    // rank 0: its key from the prompt (the lane host's), its chain, and fill_begin's sequence and spans
    const key0 = try image_key.keyIds(gpa, &prompt, &spans);
    defer gpa.free(key0);
    const h0 = pd.seed("seed");
    var k0: [3]Key = undefined;
    disk.chainKeys(h0, std.mem.sliceAsBytes(key0), &spans, &k0);
    var w: lanes_mod.Wire = .{};
    defer w.deinit(gpa);
    try w.ids(gpa, &seq0);
    try fill_spans.write(&w, gpa, &spans);
    // rank 1: the frame read back, its key ids made from it, its chain
    var r: lanes_mod.Reader = .{ .b = w.buf.items };
    var seq1: std.ArrayList(i32) = .empty;
    defer seq1.deinit(gpa);
    try r.ids(gpa, &seq1);
    var spans1: std.ArrayList(image_key.KeySpan) = .empty;
    defer spans1.deinit(gpa);
    try fill_spans.read(&r, gpa, seq1.items, &spans1);
    try r.end();
    var key1: std.ArrayList(u32) = .empty;
    defer key1.deinit(gpa);
    try fill_spans.keyIds(gpa, seq1.items, spans1.items, &key1);
    try std.testing.expectEqualSlices(u32, key0, key1.items);
    var k1: [3]Key = undefined;
    disk.chainKeys(h0, std.mem.sliceAsBytes(key1.items), spans1.items, &k1);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&k0), std.mem.sliceAsBytes(&k1));
    var text: [3]Key = undefined; // every block's key holds its pictures: none is the text chain's
    pd.chain(h0, std.mem.sliceAsBytes(&prompt), block_tokens * 4, &text);
    for (k0, text) |a, b| try std.testing.expect(!std.mem.eql(u8, &a, &b));
}
