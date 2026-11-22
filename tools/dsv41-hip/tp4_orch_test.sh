#!/bin/bash
# tp4_orch_test.sh: the four-node orchestration's checks without any box. Nothing is started, locked or reached over
# ssh: first_light.sh plan, serve_lib.sh resolution and tfp-node0 under GATE_DRY=1 only.
#   1  plan on tp4.conf: rc 0, four launch lines (P Q R S, TP2 rank / pair of each), every rank's argv carries its
#      rank and WORLD 4, rank 0 is the server
#   3  refusals at WORLD=4: TRANSPORT=tcp, three NODES, MASTER_IP not rank 0's rail address
#   4  PROMPT_DISK_GIB and LEARN_DIR at WORLD=4: the disk tier on every rank, learned states off with a line each
#   5  the gate: rankf G VERB for followers 1-3 (SERVE_RANK=G serve-rank1.sh), refusals for G 4, a rank-0 verb, a bad word
#   6  TP2 unchanged: serve.conf resolves to the same env and argv for ranks 0 and 1 with this serve_lib.sh as with the
#      base commit's (BASE, default upstream 93d28458; OLD_LIB=FILE plants another), on a stub box
#   7  first_light.sh up / down on four stub boxes: call order, all-or-nothing locks with a retry, a refused launch
set -uo pipefail
cd "$(dirname "$0")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pass=0 fail=0
ok() { echo "ok   $*"; pass=$((pass + 1)); }
no() { echo "FAIL $*"; fail=$((fail + 1)); }
check() { local what=$1; shift; if "$@"; then ok "$what"; else no "$what"; fi; }

# 1
TF_SERVE_CONF=$PWD/tp4.conf ./first_light.sh plan > "$T/plan" 2>&1; rc=$?
check "1 plan rc 0" [ "$rc" = 0 ]
check "1 four launch lines" [ "$(grep -c '^launch rank ' "$T/plan")" = 4 ]
check "1 ranks on P Q R S, TP2 rank / pair" bash -c "grep -q '^launch rank 0 on P (192.0.2.10, rail 198.51.100.10), TP2 rank 0 of pair 0' $T/plan &&
  grep -q '^launch rank 1 on Q (192.0.2.11, rail 198.51.100.11), TP2 rank 1 of pair 0' $T/plan &&
  grep -q '^launch rank 2 on R (192.0.2.12, rail 198.51.100.12), TP2 rank 0 of pair 1' $T/plan &&
  grep -q '^launch rank 3 on S (192.0.2.13, rail 198.51.100.13), TP2 rank 1 of pair 1' $T/plan"
check "1 rank 0 serves, ranks 1-3 follow with rank and world 4" bash -c "grep '^launch rank 0 ' $T/plan | grep -q 'tensorfold-native serve' &&
  for g in 1 2 3; do grep '^launch rank '\$g' ' $T/plan | grep -q \"tf-dsv41-lanes [^ ]* [^ ]*/inputs-tp4/rank\$g/rank-cache \$g 4 198.51.100.10 29620 \" || exit 1; done"
check "1 every rank on RCCL, rank 0 TF_TP_WORLD=4" bash -c "[ \$(grep '^env rank ' $T/plan | grep -c 'TF_DS_COLL=rccl') = 4 ] && grep '^env rank 0:' $T/plan | grep -q 'TF_TP_WORLD=4'"
check "1 followers carry their run dir (stop marker)" bash -c "[ \$(grep -E '^env rank [123]:' $T/plan | grep -c 'TF_TFP_RUN=') = 3 ] && ! grep '^env rank 0:' $T/plan | grep -q TF_TFP_RUN"


# 3
refused() {  # refused NAME PATTERN SED...: the conf with SED applied makes plan exit non-zero naming PATTERN
  local name=$1 pat=$2; shift 2
  sed "$@" tp4.conf > "$T/r.conf"
  TF_SERVE_CONF=$T/r.conf ./first_light.sh plan > "$T/r.out" 2>&1 && { no "$name (plan passed)"; return; }
  grep -q -- "$pat" "$T/r.out" && ok "$name" || no "$name ($(tail -1 "$T/r.out"))"
}
refused "3 WORLD=4 with TRANSPORT=tcp is refused" "needs TRANSPORT=rccl" -e 's/^TRANSPORT=rccl/TRANSPORT=tcp/' -e 's/^COLL_KIND=rccl/COLL_KIND=proxy/'
refused "3 three NODES are refused" "with 4 entries each" -e "s/^NODES=.*/NODES='P Q R'/"
refused "3 MASTER_IP off rank 0's rail address is refused" "is not rank 0's rail address" -e 's/^MASTER_IP=.*/MASTER_IP=198.51.100.11/'

# 4
sed -e 's/^PROMPT_DISK_GIB=.*/PROMPT_DISK_GIB=256/' -e 's/^PROMPT_CACHE_GIB=.*/PROMPT_CACHE_GIB=4/' tp4.conf > "$T/disk.conf"
echo "LEARN_DIR=/var/tmp/tfp-learned" >> "$T/disk.conf"
TF_SERVE_CONF=$T/disk.conf ./first_light.sh plan > "$T/plan4" 2>&1
check "4 disk tier on at WORLD=4 on every rank, learned states off with a line each" bash -c "[ \$(grep '^env rank' $T/plan4 | grep -c 'TF_DS_PROMPT_DISK_GIB=256') = 4 ] && [ \$(grep -c 'LEARN_DIR=/var/tmp/tfp-learned off at WORLD=4' $T/plan4) -ge 4 ]"

# 5
H=$T/home; mkdir -p "$H/tfp/deploy/x"
gate() { GATE_DRY=1 GATE_LOG=0 HOME=$H SSH_ORIGINAL_COMMAND="$*" ./tfp-node0; }
for g in 1 2 3; do
  check "5 rankf $g start --locked runs serve-rank1.sh as rank $g" bash -c "o=\$(GATE_DRY=1 GATE_LOG=0 HOME=$H SSH_ORIGINAL_COMMAND='tfp-node0 rankf $g start r1 tfp/deploy/x tfp/run/conf/tp4.conf --locked' ./tfp-node0) && [ \"\$o\" = 'RUN SERVE_RANK=$g $H/tfp/deploy/x/tools/dsv41-hip/serve-rank1.sh start --locked' ]"
done
for v in wait-linked mark check stop status unlock; do
  check "5 rankf 2 $v allowed" bash -c "GATE_DRY=1 GATE_LOG=0 HOME=$H SSH_ORIGINAL_COMMAND='tfp-node0 rankf 2 $v r1 tfp/deploy/x -' ./tfp-node0 | grep -q 'serve-rank1.sh $v'"
done
check "5 rankf 3 down stops tfp-lock3 units" bash -c "GATE_DRY=1 GATE_LOG=0 HOME=$H SSH_ORIGINAL_COMMAND='tfp-node0 rankf 3 down r1 tfp/deploy/x -' ./tfp-node0 | grep -q 'stop tfp-lock3 units'"
check "5 rank0 start unchanged" bash -c "[ \"\$(GATE_DRY=1 GATE_LOG=0 HOME=$H SSH_ORIGINAL_COMMAND='tfp-node0 rank0 start r1 tfp/deploy/x - --locked' ./tfp-node0)\" = 'RUN $H/tfp/deploy/x/tools/dsv41-hip/serve-rank0.sh start --locked' ]"
for c in "rankf 4 start r1 tfp/deploy/x -" "rankf 0 start r1 tfp/deploy/x -" "rankf 2 ready r1 tfp/deploy/x -" "rankf 2 bench r1 tfp/deploy/x -" \
         "rankf 2 lock r1 tfp/deploy/x - abc" "rankf 2 start r1 tfp/deploy/y/../x -" "rankf 2 start r1 tfp/deploy/x -" ; do
  if [ "$c" = "rankf 2 start r1 tfp/deploy/x -" ]; then continue; fi
  gate "tfp-node0 $c" > /dev/null 2>&1; rc=$?
  check "5 refused: $c" [ "$rc" = 126 ]
done

# 6
BASE=${BASE:-93d28458}
mkdir -p "$T/old"
if [ -n "${OLD_LIB:-}" ]; then cp "$OLD_LIB" "$T/old/serve_lib.sh"; BASE=$OLD_LIB   # (a planted base: the check must fail)
else git show "$BASE:tools/dsv41-hip/serve_lib.sh" > "$T/old/serve_lib.sh" || no "6 base serve_lib.sh at $BASE"; fi
resolve() {  # resolve LIB CONF RANK: SERVE_ENV, ----, SERVE_CMD on the stub box
  ( export HOME=$T/box; TF_SERVE_CONF=$2; source "$1"; serve_load "$3"; RUN=$T/box/run; serve_env; serve_cmd
    printf '%s\n' "${SERVE_ENV[@]}" ---- "${SERVE_CMD[@]}" ) 2>&1
}
stub() {  # stub CONF: the box files serve_load / serve_env read (RoPE tables, DEPLOY.json) under $T/box
  local dk dd rows
  dk=$(HOME=$T/box bash -c "source $1; DEPLOY_DIR=\"\$(eval echo \$DEPLOY_DIR)\"; eval echo \$DS_KIT"); dd=$(HOME=$T/box bash -c "source $1; eval echo \$DEPLOY_DIR")
  rows=$(( $(HOME=$T/box bash -c "source $1; echo \$CONTEXT") + 49 ))
  mkdir -p "$dk" "$dd"; echo '{"commit": "c0ffee"}' > "$dd/DEPLOY.json"
  for f in rope-plain-cos rope-plain-sin rope-compressed-cos rope-compressed-sin; do truncate -s $((rows * 128)) "$dk/$f.f32"; done
}
# serve.conf's one deliberate argv change since the base is LOOP_POLICY (--loop-policy stop): compared with it unset,
# which resolves to the base's LOOP_GUARD flag
sed 's/^LOOP_POLICY=.*/LOOP_POLICY=/' "$PWD/serve.conf" > "$T/serve-nopolicy.conf"
for conf in "$T/serve-nopolicy.conf"; do
  [ -f "$conf" ] || { echo "skip 6 $conf (not here)"; continue; }
  rm -rf "$T/box"; stub "$conf"
  for r in 0 1; do
    # (TF_GPU_COUNTS=0 and TF_DS_PLACE_VICTIM=0 are passed explicitly here: both are the engine's off default)
    resolve "$T/old/serve_lib.sh" "$conf" "$r" > "$T/old.$r"
    resolve "$PWD/serve_lib.sh" "$conf" "$r" | grep -vx 'TF_GPU_COUNTS=0\|TF_DS_PLACE_VICTIM=0' > "$T/new.$r"
    check "6 $(basename "$conf") rank $r: same env and argv as $BASE ($(grep -c . "$T/new.$r") lines)" bash -c "grep -q -- '^----$' $T/new.$r && cmp -s $T/old.$r $T/new.$r || { diff $T/old.$r $T/new.$r | head -5; exit 1; }"
  done
done

# 7: first_light.sh up and down on four boxes against stubs (ssh and this box's rank 1): the call order, the GPU locks
# all or nothing (rank 3's lock busy once: every lock taken is released, the latest first, and the set retried), the
# followers before rank 0, every follower linked; down marks ranks 2 and 3 before rank 0 stops and takes each down
H7=$T/h7; mkdir -p "$H7/tfp/run/conf" "$T/bin"
sed 's/^BENCH=1/BENCH=0/' tp4.conf > "$H7/tfp/run/conf/t7.conf"
cat > "$T/bin/ssh" <<'SH'
#!/bin/bash
# stub: ssh HOST GATE VERB...: one normalized line a call in $CALLS, canned answers
n=$(case ${1##*.} in 10) echo P;; 11) echo Q;; 12) echo R;; 13) echo S;; *) echo "?$1";; esac); shift 2
js() { echo "{\"rank\": $1, \"host\": \"h${HOSTOVERRIDE:-$2}\", \"deploy\": {\"commit\": \"c0\"}, \"kit_manifest_sha256\": \"k\", \"config_sha256\": \"c\", \"env\": {\"TF_DS_COLL\": \"rccl\", \"TF_COLL_KIND\": \"rccl\", \"NCCL_IB_GID_INDEX\": \"3\"}}"; }
case $1 in
  uptime|quiet) echo "$n $*" >> "$CALLS"; echo up;;
  rank0) echo "$n rank0 $2${6:+ $6}" >> "$CALLS"
         case $2 in check) js 0 "$n";; ready) echo '{"reply": "Paris"}';; status) echo "rank 0 not running: x";; esac;;
  rankf) echo "$n rankf $2 $3${7:+ $7}" >> "$CALLS"
         case $3 in check) js "$2" "$n";; status) echo "rank $2 not running: x";;
           lock) [ "$2" = 3 ] && [ -e "$BUSY3" ] && { rm -f "$BUSY3"; exit 1; };; esac;;
esac
exit 0
SH
cat > "$T/r1" <<'SH'
#!/bin/bash
echo "Q local $*" >> "$CALLS"
R=$HOME/tfp/run/tp4/$RUN_ID; mkdir -p "$R"
[ "$1" = check ] && echo '{"rank": 1, "host": "hQ", "deploy": {"commit": "c0"}, "kit_manifest_sha256": "k", "config_sha256": "c", "env": {"TF_DS_COLL": "rccl", "TF_COLL_KIND": "rccl", "NCCL_IB_GID_INDEX": "3"}}' > "$R/resolved-rank1.json"
exit 0
SH
chmod +x "$T/bin/ssh" "$T/r1"
fl7() { env HOME="$H7" PATH="$T/bin:$PATH" CALLS="$T/calls" BUSY3="$T/busy3" FL_RANK1="$T/r1" SERVE_PLAN=1 LOCK_RETRY_S=0 \
  RUN_ID=t7 TF_SERVE_CONF="$H7/tfp/run/conf/t7.conf" ./first_light.sh "$@"; }
: > "$T/calls"; touch "$T/busy3"
fl7 up > "$T/up7" 2>&1; rc=$?
cat > "$T/want-up" <<'W'
P rank0 check
Q local check
R rankf 2 check
S rankf 3 check
P rank0 lock
Q local lock 60
R rankf 2 lock 60
S rankf 3 lock 60
R rankf 2 unlock
Q local unlock
P rank0 unlock
P rank0 lock
Q local lock 60
R rankf 2 lock 60
S rankf 3 lock 60
Q local start --locked
R rankf 2 start --locked
S rankf 3 start --locked
P rank0 start --locked
P rank0 ready
Q local wait-linked
R rankf 2 wait-linked
S rankf 3 wait-linked
P uptime
R uptime
S uptime
W
check "7 up: rc 0, all four checked, locks all or nothing with one retry, followers first, all linked" bash -c "[ $rc = 0 ] && diff $T/want-up $T/calls"
check "7 up: the four identities compared" grep -q '^all 4 ranks (hP, hQ, hR, hS): .*collectives rccl' "$T/up7"
check "7 up: first_light.json names world 4 and every rank" python3 -c "import json,sys; j=json.load(open('$H7/tfp/run/tp4/t7/first_light.json')); sys.exit(0 if j['world']==4 and all(j[f'rank{g}']['rank']==g for g in range(4)) else 1)"
: > "$T/calls"
fl7 down > "$T/down7" 2>&1; rc=$?
printf '%s\n' "R rankf 2 mark" "S rankf 3 mark" "P rank0 stop" "Q local stop" "Q local unlock" "R rankf 2 down" "S rankf 3 down" "P rank0 down" > "$T/want-down"
check "7 down: followers marked before rank 0 stops, each taken down" bash -c "[ $rc = 0 ] && diff $T/want-down $T/calls && [ -e $H7/tfp/run/tp4/t7/stopping ]"
: > "$T/calls"
HOSTOVERRIDE=P fl7 up > "$T/up7b" 2>&1; rc=$?
check "7 up refuses four ranks that are not four boxes, before any lock" bash -c "[ $rc != 0 ] && grep -q 'different boxes' $T/up7b && ! grep -q lock $T/calls"

echo "tp4_orch_test: $pass passed, $fail failed"
[ "$fail" = 0 ]
