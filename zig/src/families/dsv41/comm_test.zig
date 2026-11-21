//! the RCCL collectives: the family's NCCL Comm on HIP (comm.zig over gpu/hip/rccl.zig), with no GPU work:
//! - the real librccl (TF_RCCL_LIB; zig build test-dsv41-hip-coll sets it) opens through the facade with every entry
//!   point, reports its version, and leaves one HIP runtime in the process;
//! - four ranks on threads link over loopback Fans as the engine does (rank 0 listens on PORT + g - 1), share rank 0's
//!   unique id through Comm.fromFanWith, and run Comm.coll's all-gathers and the 2D split's exchanges against a host
//!   fake of the RCCL calls (pointers are host addresses, the stream is unused): every byte lands in rank order, a
//!   failed call poisons the Coll, abort turns every later call into Aborted.
const std = @import("std");
const gpu = @import("gpu");
const comm_mod = @import("comm.zig");
const coll = @import("coll.zig");
const Fan = @import("fan.zig").Fan;
const nccl = gpu.nccl;

const world = 4;
const Atomic = std.atomic.Value(u32);

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// The host fake of RCCL's calls for `world` ranks of one process.
const Fake = struct {
    var id: nccl.UniqueId = undefined;
    var ids_made: Atomic = .init(0);
    var calls: [world]Atomic = .{ Atomic.init(0), Atomic.init(0), Atomic.init(0), Atomic.init(0) };
    /// a rank's n-th collective call fails with ncclSystemError (0: none); every rank fails the same call
    var fail_at: [world]u32 = .{ 0, 0, 0, 0 };
    var slot_ptr: [world]u64 = undefined;
    var slot_len: [world]usize = undefined;
    var arrived: Atomic = .init(0);
    var sense: Atomic = .init(0);
    var post_ptr: [world][world]u64 = undefined;
    var post_len: [world][world]usize = undefined;
    /// post_flag[from][to]: 0 empty, 1 posted by `from`, 2 taken by `to`
    var post_flag: [world][world]Atomic = @splat(@splat(Atomic.init(0)));

    const Op = struct { send: bool, ptr: u64, bytes: usize, peer: u32, rank: u32 };
    threadlocal var group: [8]Op = undefined;
    threadlocal var group_n: usize = 0;
    threadlocal var in_group: bool = false;

    const api: nccl.Api = .{
        .ncclGetVersion = getVersion,
        .ncclGetUniqueId = getUniqueId,
        .ncclCommInitRank = commInitRank,
        .ncclCommInitAll = commInitAll,
        .ncclCommDestroy = commDestroy,
        .ncclGetErrorString = errorString,
        .ncclAllReduce = allReduce,
        .ncclAllGather = allGather,
        .ncclBroadcast = broadcast,
        .ncclSend = send,
        .ncclRecv = recv,
        .ncclGroupStart = groupStart,
        .ncclGroupEnd = groupEnd,
    };

    const S = gpu.abi.Stream;
    const invalid_argument = 4;
    const invalid_usage = 5;
    const system_error = 2;

    fn rankOf(c: nccl.Comm) u32 {
        return @intCast(@intFromPtr(c.?) - 1);
    }

    fn spin(flag: *Atomic, want: u32) void {
        while (flag.load(.acquire) != want) std.atomic.spinLoopHint();
    }

    fn barrier() void {
        const s = sense.load(.acquire);
        if (arrived.fetchAdd(1, .acq_rel) == world - 1) {
            arrived.store(0, .release);
            sense.store(s +% 1, .release);
        } else spin(&sense, s +% 1);
    }

    fn failNow(rank: u32) bool {
        const n = calls[rank].fetchAdd(1, .acq_rel) + 1;
        return fail_at[rank] == n;
    }

    fn getVersion(v: *c_int) callconv(.c) c_int {
        v.* = 22803;
        return 0;
    }

    fn getUniqueId(out: *nccl.UniqueId) callconv(.c) c_int {
        const k = ids_made.fetchAdd(1, .acq_rel);
        for (&id.internal, 0..) |*b, i| b.* = @truncate(i *% 131 +% 7 +% k *% 29);
        out.* = id;
        return 0;
    }

    fn commInitRank(c: *nccl.Comm, n: c_int, got: nccl.UniqueId, rank: c_int) callconv(.c) c_int {
        if (n != world or rank < 0 or rank >= world) return invalid_argument;
        if (!std.mem.eql(u8, &got.internal, &id.internal)) return invalid_usage; // not rank 0's latest id
        c.* = @ptrFromInt(@as(usize, @intCast(rank)) + 1);
        return 0;
    }

    fn commInitAll(_: [*]nccl.Comm, _: c_int, _: ?[*]const c_int) callconv(.c) c_int {
        return invalid_usage;
    }

    fn commDestroy(_: nccl.Comm) callconv(.c) c_int {
        return 0;
    }

    fn errorString(r: c_int) callconv(.c) ?[*:0]const u8 {
        return if (r == system_error) "fake: unhandled system error" else "fake: invalid call";
    }

    fn allReduce(_: u64, _: u64, _: usize, _: nccl.DataType, _: nccl.RedOp, _: nccl.Comm, _: S) callconv(.c) c_int {
        return invalid_usage;
    }

    fn broadcast(_: u64, _: u64, _: usize, _: nccl.DataType, _: c_int, _: nccl.Comm, _: S) callconv(.c) c_int {
        return invalid_usage;
    }

    fn allGather(src: u64, dst: u64, count: usize, dt: nccl.DataType, c: nccl.Comm, _: S) callconv(.c) c_int {
        const r = rankOf(c);
        if (dt != .u8) return invalid_argument;
        if (failNow(r)) return system_error;
        slot_ptr[r] = src;
        slot_len[r] = count;
        barrier();
        var rc: c_int = 0;
        const out: [*]u8 = @ptrFromInt(dst);
        for (0..world) |g| {
            if (slot_len[g] != count) {
                rc = invalid_usage;
                continue;
            }
            const in: [*]const u8 = @ptrFromInt(slot_ptr[g]);
            @memcpy(out[g * count ..][0..count], in[0..count]);
        }
        barrier();
        return rc;
    }

    fn groupStart() callconv(.c) c_int {
        in_group = true;
        group_n = 0;
        return 0;
    }

    fn queue(is_send: bool, buf: u64, count: usize, dt: nccl.DataType, peer: c_int, c: nccl.Comm) c_int {
        if (!in_group or group_n == group.len or dt != .u8 or peer < 0 or peer >= world) return invalid_usage;
        group[group_n] = .{ .send = is_send, .ptr = buf, .bytes = count, .peer = @intCast(peer), .rank = rankOf(c) };
        group_n += 1;
        return 0;
    }

    fn send(buf: u64, count: usize, dt: nccl.DataType, peer: c_int, c: nccl.Comm, _: S) callconv(.c) c_int {
        return queue(true, buf, count, dt, peer, c);
    }

    fn recv(buf: u64, count: usize, dt: nccl.DataType, peer: c_int, c: nccl.Comm, _: S) callconv(.c) c_int {
        return queue(false, buf, count, dt, peer, c);
    }

    fn groupEnd() callconv(.c) c_int {
        if (!in_group) return invalid_usage;
        in_group = false;
        const ops = group[0..group_n];
        if (ops.len == 0) return 0;
        const r = ops[0].rank;
        if (failNow(r)) return system_error;
        for (ops) |o| if (o.send) {
            post_ptr[r][o.peer] = o.ptr;
            post_len[r][o.peer] = o.bytes;
            post_flag[r][o.peer].store(1, .release);
        };
        var rc: c_int = 0;
        for (ops) |o| if (!o.send) {
            spin(&post_flag[o.peer][r], 1);
            if (post_len[o.peer][r] != o.bytes) {
                rc = invalid_usage;
            } else {
                const in: [*]const u8 = @ptrFromInt(post_ptr[o.peer][r]);
                const out: [*]u8 = @ptrFromInt(o.ptr);
                @memcpy(out[0..o.bytes], in[0..o.bytes]);
            }
            post_flag[o.peer][r].store(2, .release);
        };
        for (ops) |o| if (o.send) {
            spin(&post_flag[r][o.peer], 2);
            post_flag[r][o.peer].store(0, .release);
        };
        return rc;
    }
};

fn gatherByte(g: usize, i: usize, size: usize) u8 {
    return @truncate(g *% 131 +% i *% 7 +% size);
}

fn exchangeBytes(from: u32, to: u32) usize {
    return 1000 + 17 * @as(usize, from) + 5 * @as(usize, to);
}

fn exchangeByte(from: u32, to: u32, i: usize) u8 {
    return @truncate(@as(usize, from) *% 31 +% @as(usize, to) *% 7 +% i *% 13);
}

const Rank = struct {
    g: u32,
    port: u16,
    err: ?anyerror = null,
    gather_ns: u64 = 0,
    exchange_ns: u64 = 0,
    reps: usize = 0,

    fn run(r: *Rank) void {
        r.body() catch |e| {
            r.err = e;
        };
    }

    fn body(r: *Rank) !void {
        const a = std.heap.page_allocator;
        const g = r.g;
        var fan = try Fan.open(std.testing.io, .{ 127, 0, 0, 1 }, r.port, g, world);
        defer fan.close();
        const stream: gpu.Stream = .{ .d = undefined, .handle = null };
        const site = coll.Site.at(.test_only, 0);

        // 1. all-gathers in rank order and every peer's exchange, through Comm.coll
        var c1 = try comm_mod.Comm.fromFanWith(nccl.Library.fake(Fake.api), &fan, g, world);
        var co = c1.coll();
        try std.testing.expect(co.capturable);
        try std.testing.expectEqual(@as(u32, world), co.world);
        for ([_]usize{ 1, 4096, 5120 * 4 * 6 + 3, 1 << 20 }) |size| {
            const src = try a.alloc(u8, size);
            defer a.free(src);
            const dst = try a.alloc(u8, world * size);
            defer a.free(dst);
            for (src, 0..) |*b, i| b.* = gatherByte(g, i, size);
            @memset(dst, 0);
            try co.allGather(site, @intFromPtr(src.ptr), @intFromPtr(dst.ptr), size, .u8, stream);
            for (0..world) |h| for (0..size) |i| if (dst[h * size + i] != gatherByte(h, i, size)) return error.GatherWrongByte;
        }
        // f32 counts become bytes
        {
            var src: [6]f32 = undefined;
            var dst: [world * 6]f32 = undefined;
            for (&src, 0..) |*x, i| x.* = @floatFromInt(g * 100 + i);
            try co.allGather(site, @intFromPtr(&src), @intFromPtr(&dst), 6, .f32, stream);
            for (0..world) |h| for (0..6) |i| if (dst[h * 6 + i] != @as(f32, @floatFromInt(h * 100 + i))) return error.GatherWrongFloat;
        }
        // the 2D split's exchanges: the pair partner (g ^ 1), the other pair's same rank (g ^ 2), and g ^ 3; sizes differ
        // by direction
        for ([_]u32{ 1, 2, 3, 2, 1 }) |k| {
            const peer = g ^ k;
            const sb = exchangeBytes(g, peer);
            const rb = exchangeBytes(peer, g);
            const src = try a.alloc(u8, sb);
            defer a.free(src);
            const dst = try a.alloc(u8, rb);
            defer a.free(dst);
            for (src, 0..) |*b, i| b.* = exchangeByte(g, peer, i);
            @memset(dst, 0);
            try co.exchange(site, @intFromPtr(src.ptr), sb, @intFromPtr(dst.ptr), rb, .u8, peer, stream);
            for (dst, 0..) |b, i| if (b != exchangeByte(peer, g, i)) return error.ExchangeWrongByte;
        }
        // the facade's own cost per call on this host (fake transport: a four-thread barrier and the copies; no RCCL
        // kernel, no wire)
        {
            var src: [4096]u8 = @splat(@truncate(g));
            var dst: [world * 4096]u8 = undefined;
            const reps = 2000;
            for (0..100) |_| try co.allGather(site, @intFromPtr(&src), @intFromPtr(&dst), 4096, .u8, stream);
            const t0 = nowNs();
            for (0..reps) |_| try co.allGather(site, @intFromPtr(&src), @intFromPtr(&dst), 4096, .u8, stream);
            const t1 = nowNs();
            const peer = g ^ 1;
            for (0..reps) |_| try co.exchange(site, @intFromPtr(&src), 4096, @intFromPtr(&dst), 4096, .u8, peer, stream);
            const t2 = nowNs();
            r.gather_ns = (t1 - t0) / reps;
            r.exchange_ns = (t2 - t1) / reps;
            r.reps = reps;
        }
        co.close();

        // 2. a failed call poisons the Coll: the same error after it, and the transport is not called again
        var c2 = try comm_mod.Comm.fromFanWith(nccl.Library.fake(Fake.api), &fan, g, world);
        var co2 = c2.coll();
        defer co2.close();
        var one: [8]u8 = @splat(@truncate(g));
        var all: [world * 8]u8 = undefined;
        Fake.fail_at[g] = Fake.calls[g].load(.acquire) + 2;
        try co2.allGather(site, @intFromPtr(&one), @intFromPtr(&all), 8, .u8, stream);
        try std.testing.expectError(error.CollectiveFailed, co2.allGather(site, @intFromPtr(&one), @intFromPtr(&all), 8, .u8, stream));
        const after = Fake.calls[g].load(.acquire);
        try std.testing.expectError(error.CollectiveFailed, co2.exchange(site, @intFromPtr(&one), 8, @intFromPtr(&all), 8, .u8, g ^ 1, stream));
        try std.testing.expectError(error.CollectiveFailed, co2.allGather(site, @intFromPtr(&one), @intFromPtr(&all), 8, .u8, stream));
        try std.testing.expectEqual(after, Fake.calls[g].load(.acquire));

        // 3. abort: every later call is Aborted without reaching the transport; idempotent
        var c3 = try comm_mod.Comm.fromFanWith(nccl.Library.fake(Fake.api), &fan, g, world);
        var co3 = c3.coll();
        defer co3.close();
        co3.abort(.cancelled);
        co3.abort(.shutdown);
        const before = Fake.calls[g].load(.acquire);
        try std.testing.expectError(error.Aborted, co3.allGather(site, @intFromPtr(&one), @intFromPtr(&all), 8, .u8, stream));
        try std.testing.expectError(error.Aborted, co3.exchange(site, @intFromPtr(&one), 8, @intFromPtr(&all), 8, .u8, g ^ 2, stream));
        try std.testing.expectEqual(before, Fake.calls[g].load(.acquire));
    }
};

test "librccl opens through the HIP facade: every entry point, its version, one HIP runtime" {
    const path = std.c.getenv("TF_RCCL_LIB") orelse return error.SkipZigTest;
    var lib = try nccl.Library.openPath(std.mem.span(path));
    defer lib.close();
    const v = try lib.version();
    var rt: nccl.Runtimes = .{};
    rt.scan();
    std.debug.print("librccl loaded: {s} version {d}; HIP runtimes in the process: {d} ({s})\n", .{ path, v, rt.count, rt.name(0) });
    try std.testing.expect(v >= 22800);
    try std.testing.expectEqual(@as(usize, 1), rt.count);
}

test "four ranks over loopback Fans: rank 0's id, all-gathers in rank order and the 2D split's exchanges through Comm.coll" {
    const port: u16 = 31000 + @as(u16, @intCast(@mod(std.c.getpid(), 4000))) * 4;
    var ranks: [world]Rank = undefined;
    var threads: [world]std.Thread = undefined;
    for (&ranks, 0..) |*r, g| r.* = .{ .g = @intCast(g), .port = port };
    for (&threads, &ranks) |*t, *r| t.* = try std.Thread.spawn(.{}, Rank.run, .{r});
    for (threads) |t| t.join();
    for (ranks) |r| if (r.err) |e| {
        std.debug.print("rank {d}: {s}\n", .{ r.g, @errorName(e) });
        return e;
    };
    try std.testing.expectEqual(@as(u32, 3), Fake.ids_made.load(.acquire)); // one id per Comm, made by rank 0 only
    for (ranks) |r| std.debug.print("rank {d}: fake 4 KiB all_gather {d} ns/call, 4 KiB exchange {d} ns/call ({d} reps; host fake, no RCCL kernel, no wire)\n", .{ r.g, r.gather_ns, r.exchange_ns, r.reps });
}
