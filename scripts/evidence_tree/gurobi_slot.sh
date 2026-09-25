#!/usr/bin/env bash
# Gurobi slot guard: keep at most GUROBI_MAX concurrent Gurobi jobs per account.
#
# Gurobi licenses are token-limited; launching more concurrent MILP jobs than
# the license allows gets jobs killed.  This guard makes a job WAIT for a free
# slot instead of dying, so extra tasks queue rather than fail.
#
#   source scripts/evidence_tree/gurobi_slot.sh
#   gurobi_slot_acquire        # blocks until a slot is free (or reclaims a stale one)
#   ... run Gurobi work ...
#   # released automatically on EXIT/INT/TERM via trap
#
# Slots are directories created with mkdir (atomic on local FS and NFS).  A slot
# is reclaimable when its recorded PID is no longer alive, so a job killed by
# PBS or the scheduler does not deadlock the queue.
GUROBI_SLOTS_DIR="${GUROBI_SLOTS_DIR:-${HOME}/.gurobi_slots}"
GUROBI_MAX="${GUROBI_MAX:-2}"
GUROBI_WAIT_SECONDS="${GUROBI_WAIT_SECONDS:-30}"

gurobi_slot_acquire() {
  mkdir -p "${GUROBI_SLOTS_DIR}"
  local i d pid waited=0
  while :; do
    for ((i = 1; i <= GUROBI_MAX; i++)); do
      d="${GUROBI_SLOTS_DIR}/slot_${i}"
      if mkdir "${d}" 2>/dev/null; then
        printf '%s\n' "$$" > "${d}/pid"
        date -u +%FT%TZ > "${d}/since" 2>/dev/null || true
        GUROBI_SLOT="${d}"
        echo "[gurobi_slot] acquired slot ${i}/${GUROBI_MAX} (pid $$)"
        return 0
      fi
      # Reclaim a stale slot whose owner is gone.
      pid="$(cat "${d}/pid" 2>/dev/null || true)"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        echo "[gurobi_slot] reclaiming stale slot ${i} (dead pid ${pid})"
        rm -rf "${d}"
      fi
    done
    if (( waited % 300 == 0 )); then
      echo "[gurobi_slot] all ${GUROBI_MAX} slots busy; queued (waited ${waited}s)"
    fi
    sleep "${GUROBI_WAIT_SECONDS}"
    waited=$((waited + GUROBI_WAIT_SECONDS))
  done
}

gurobi_slot_release() {
  if [[ -n "${GUROBI_SLOT:-}" && -d "${GUROBI_SLOT}" ]]; then
    rm -rf "${GUROBI_SLOT}"
    echo "[gurobi_slot] released ${GUROBI_SLOT}"
    GUROBI_SLOT=""
  fi
}

# Release on normal exit, and on INT/TERM release AND exit so a scheduler kill
# (or `timeout`) actually stops the wait loop instead of being swallowed.
trap 'gurobi_slot_release' EXIT
trap 'gurobi_slot_release; exit 143' INT TERM
