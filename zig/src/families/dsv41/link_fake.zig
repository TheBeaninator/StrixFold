//! link_fake: an in-process collective over up to eight ranks for host tests, with no socket and no GPU. Each rank is
//! a thread of the test and its device addresses are host addresses. A call returns once every rank has made the same
//! one (kind, site and sizes compared, as the wire header does; Mismatch on every rank otherwise) with its recv
//! written: an all-gather puts every rank's send in rank order, an exchange the peer's send. The four-node split's
//! exchanges run over it in prompt2d_test.zig; later world-4 host tests can use it the same way.
const std = @import("std");
const gpu = @import("gpu");
const coll = @import("coll.zig");

pub const max_world = 8;

const Call = struct {
    kind: enum { gather, exchange } = .gather,
    site: coll.Site = .at(.test_only, 0),
    send: u64 = 0,
    bytes: usize = 0,
    recv: u64 = 0,
    rbytes: usize = 0,
    peer: u32 = 0,
};

/// The ranks' shared state: a barrier and each rank's posted call.
pub const Board = struct {
    world: u32,
    arrived: std.atomic.Value(u32) = .init(0),
    phase: std.atomic.Value(u32) = .init(0),
    calls: [max_world]Call = @splat(.{}),

    pub fn init(world: u32) Board {
        std.debug.assert(world >= 1 and world <= max_world);
        return .{ .world = world };
    }

    /// Every rank's arrival (sense-reversing: the last one in resets the count and releases the rest).
    fn wait(b: *Board) void {
        const ph = b.phase.load(.acquire);
        if (b.arrived.fetchAdd(1, .acq_rel) + 1 == b.world) {
            b.arrived.store(0, .monotonic);
            b.phase.store(ph +% 1, .release);
        } else while (b.phase.load(.acquire) == ph) std.atomic.spinLoopHint();
    }

    /// Whether the posted calls make one collective: the same kind and site, a gather's bytes alike, every exchange's
    /// peer naming it back with matching sizes. Every rank reads the same board, so all agree on the answer.
    fn agree(b: *const Board) bool {
        const c0 = b.calls[0];
        for (b.calls[0..b.world], 0..) |c, g| {
            if (c.kind != c0.kind or c.site.phase != c0.site.phase or c.site.layer != c0.site.layer) return false;
            switch (c.kind) {
                .gather => if (c.bytes != c0.bytes) return false,
                .exchange => {
                    if (c.peer >= b.world or c.peer == g) return false;
                    const q = b.calls[c.peer];
                    if (q.kind != .exchange or q.peer != g or q.bytes != c.rbytes or q.rbytes != c.bytes) return false;
                },
            }
        }
        return true;
    }
};

fn move(src: u64, dst: u64, bytes: usize) void {
    if (bytes == 0 or src == dst) return; // an in-place gather's own block
    const s: [*]const u8 = @ptrFromInt(src);
    const d: [*]u8 = @ptrFromInt(dst);
    @memcpy(d[0..bytes], s[0..bytes]);
}

/// One rank's end: handle() is its Coll (exchange included, not capturable).
pub const Rank = struct {
    b: *Board,
    rank: u32,
    gathers: u64 = 0,
    exchanges: u64 = 0,

    pub fn handle(r: *Rank) coll.Coll {
        return .{ .ptr = r, .vtable = &vtable, .rank = r.rank, .world = r.b.world, .capturable = false };
    }

    const vtable: coll.Coll.VTable = .{ .all_gather = allGather, .exchange = exchange, .abort = abort, .close = close };

    fn post(r: *Rank, c: Call) coll.Error!void {
        const b = r.b;
        b.calls[r.rank] = c;
        b.wait();
        const ok = b.agree();
        if (ok) switch (c.kind) {
            .gather => for (0..b.world) |g| move(b.calls[g].send, c.recv + g * c.bytes, c.bytes),
            .exchange => move(b.calls[c.peer].send, c.recv, c.rbytes),
        };
        b.wait(); // no rank posts again, or reuses its send, before every rank has read the board
        if (!ok) return error.Mismatch;
        switch (c.kind) {
            .gather => r.gathers += 1,
            .exchange => r.exchanges += 1,
        }
    }

    fn allGather(ptr: *anyopaque, _: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const r: *Rank = @ptrCast(@alignCast(ptr));
        return r.post(.{ .kind = .gather, .site = site, .send = send, .bytes = bytes, .recv = recv });
    }

    fn exchange(ptr: *anyopaque, _: gpu.Stream, site: coll.Site, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, peer: u32) coll.Error!void {
        const r: *Rank = @ptrCast(@alignCast(ptr));
        return r.post(.{ .kind = .exchange, .site = site, .send = send, .bytes = send_bytes, .recv = recv, .rbytes = recv_bytes, .peer = peer });
    }

    // nothing here blocks on a lost peer: abort and close have nothing to cut short
    fn abort(_: *anyopaque, _: coll.Reason) void {}
    fn close(_: *anyopaque) void {}
};

const SelfTest = struct {
    rank: Rank,
    buf: [4 * 16]u8 align(16) = undefined,
    got: [3]?coll.Error = .{ null, null, null },

    fn run(t: *SelfTest) void {
        const c = t.rank.handle();
        const g: u8 = @intCast(t.rank.rank);
        var send: [16]u8 = @splat(g);
        var recv: [16]u8 = undefined;
        const s: gpu.Stream = undefined;
        t.got[0] = if (c.allGather(.at(.test_only, 1), @intFromPtr(&send), @intFromPtr(&t.buf), 16, .u8, s)) null else |e| e;
        // rank 3 asks for a different size: every rank sees Mismatch, and the board still serves the next call
        const n: usize = if (g == 3) 8 else 16;
        t.got[1] = if (c.allGather(.at(.test_only, 2), @intFromPtr(&send), @intFromPtr(&t.buf), n, .u8, s)) null else |e| e;
        t.got[2] = if (c.exchange(.at(.test_only, 3), @intFromPtr(&send), 16, @intFromPtr(&recv), 16, .u8, g ^ 2, s)) null else |e| switch (e) {
            error.Unsupported => error.CollectiveFailed,
            else => |x| x,
        };
        if (t.got[2] == null and recv[0] != (g ^ 2)) t.got[2] = error.CollectiveFailed;
    }
};

test "link_fake: four ranks gather in rank order, exchange with a peer, and all see a mismatched call" {
    var b: Board = .init(4);
    var ts: [4]SelfTest = undefined;
    var th: [4]std.Thread = undefined;
    for (&ts, &th, 0..) |*t, *h, g| {
        t.* = .{ .rank = .{ .b = &b, .rank = @intCast(g) } };
        h.* = try std.Thread.spawn(.{}, SelfTest.run, .{t});
    }
    for (th) |h| h.join();
    for (ts) |t| {
        try std.testing.expectEqual(@as(?coll.Error, null), t.got[0]);
        for (0..4) |g| try std.testing.expect(std.mem.allEqual(u8, t.buf[g * 16 ..][0..16], @intCast(g)));
        try std.testing.expectEqual(@as(?coll.Error, error.Mismatch), t.got[1]);
        try std.testing.expectEqual(@as(?coll.Error, null), t.got[2]);
        try std.testing.expectEqual(@as(u64, 1), t.rank.gathers);
        try std.testing.expectEqual(@as(u64, 1), t.rank.exchanges);
    }
}


/// TF_DS_COLL_TIME's cost: link_fake calls a rank per second with the timer off and on
/// (call mode: link_fake has no stream to drain), half gathers half pair exchanges, at world 2 and 4, best of 3 runs
/// alternating off and on; then the wrapper alone on a transport that does nothing (ns a call). TF_COLL_TIMER_BENCH=N
/// sets the calls a run (default 2000, a quick check); the numbers go to stderr and, with TF_COLL_TIMER_BENCH_OUT set,
/// are appended to that file.
const TimerBench = struct {
    rank: Rank,
    timer: coll.Timer = .{ .mode = .call },
    on: bool,
    n: usize,

    fn run(b: *TimerBench) void {
        var c = b.rank.handle();
        if (b.on) c.timer = &b.timer;
        var send: [64]u8 = @splat(1);
        var recv: [max_world * 64]u8 = undefined;
        const s: gpu.Stream = undefined;
        for (0..b.n) |i| {
            if (i % 2 == 0)
                c.allGather(.at(.round_attn, 0), @intFromPtr(&send), @intFromPtr(&recv), 64, .u8, s) catch unreachable
            else
                c.exchange(.at(.two_pair, 0), @intFromPtr(&send), 64, @intFromPtr(&recv), 64, .u8, b.rank.rank ^ 1, s) catch unreachable;
        }
    }

    /// One run of `world` ranks: calls a rank per second.
    fn rate(world: u32, on: bool, n: usize, timers: *[max_world]coll.Timer) !f64 {
        var board: Board = .init(world);
        var bs: [max_world]TimerBench = undefined;
        var th: [max_world]std.Thread = undefined;
        const t0 = coll.Timer.nowNs();
        for (0..world) |g| {
            bs[g] = .{ .rank = .{ .b = &board, .rank = @intCast(g) }, .on = on, .n = n };
            th[g] = try std.Thread.spawn(.{}, run, .{&bs[g]});
        }
        for (th[0..world]) |h| h.join();
        const dt = coll.Timer.nowNs() - t0;
        for (0..world) |g| timers[g] = bs[g].timer;
        return @as(f64, @floatFromInt(n)) * std.time.ns_per_s / @as(f64, @floatFromInt(dt));
    }
};

const Nop = struct {
    fn gather(_: *anyopaque, _: gpu.Stream, _: coll.Site, _: u64, _: u64, _: usize) coll.Error!void {}
    fn abort(_: *anyopaque, _: coll.Reason) void {}
    fn close(_: *anyopaque) void {}
    const vt: coll.Coll.VTable = .{ .all_gather = gather, .abort = abort, .close = close };
};

fn nopNs(on: bool, n: usize) u64 {
    var x: u8 = 0;
    var t: coll.Timer = .{ .mode = .call };
    var c: coll.Coll = .{ .ptr = &x, .vtable = &Nop.vt, .rank = 0, .world = 2, .capturable = false };
    if (on) c.timer = &t;
    const t0 = coll.Timer.nowNs();
    for (0..n) |_| c.allGather(.at(.round_attn, 0), 0, 0, 16, .u8, undefined) catch unreachable;
    return coll.Timer.nowNs() - t0;
}

test "TF_DS_COLL_TIME on link_fake: every call counted on every rank, and its cost (calls a second, off vs on)" {
    const n: usize = if (std.c.getenv("TF_COLL_TIMER_BENCH")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 2000) else 2000;
    var out: [1024]u8 = undefined;
    var w: usize = 0;
    for ([_]u32{ 2, 4 }) |world| {
        var best: [2]f64 = .{ 0, 0 };
        for (0..3) |_| for ([_]bool{ false, true }) |on| {
            var timers: [max_world]coll.Timer = undefined;
            const r = try TimerBench.rate(world, on, n, &timers);
            best[@intFromBool(on)] = @max(best[@intFromBool(on)], r);
            if (on) for (timers[0..world]) |t| {
                try std.testing.expectEqual(@as(u64, (n + 1) / 2), t.rounds.calls[0]);
                try std.testing.expectEqual(@as(u64, n / 2), t.rounds.calls[1]);
                try std.testing.expectEqual(@as(u64, 0), t.pass.calls[0] + t.pass.calls[1] + t.pass.calls[2]);
            };
        };
        w += (std.fmt.bufPrint(out[w..], "coll timer link_fake world {d}, {d} calls a run, best of 3: off {d:.0} calls/s, on {d:.0} calls/s ({d:.2}%)\n", .{
            world, n, best[0], best[1], (best[0] - best[1]) / best[0] * 100,
        }) catch out[w..w]).len;
    }
    const m = n * 100;
    var nop: [2]u64 = .{ std.math.maxInt(u64), std.math.maxInt(u64) };
    for (0..3) |_| for ([_]bool{ false, true }) |on| {
        nop[@intFromBool(on)] = @min(nop[@intFromBool(on)], nopNs(on, m));
    };
    w += (std.fmt.bufPrint(out[w..], "coll timer wrapper alone ({d} calls, best of 3): off {d:.1} ns a call, on {d:.1} ns a call\n", .{
        m, @as(f64, @floatFromInt(nop[0])) / @as(f64, @floatFromInt(m)), @as(f64, @floatFromInt(nop[1])) / @as(f64, @floatFromInt(m)),
    }) catch out[w..w]).len;
    std.debug.print("{s}", .{out[0..w]});
    if (std.c.getenv("TF_COLL_TIMER_BENCH_OUT")) |path| {
        const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
        if (fd >= 0) {
            _ = std.c.write(fd, &out, w);
            _ = std.c.close(fd);
        }
    }
}
