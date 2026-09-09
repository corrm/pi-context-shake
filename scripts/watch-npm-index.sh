#!/usr/bin/env bash
# Poll the npm search index until pi-context-shake appears in keywords:pi-package.
# Exit 0 when found, 1 when giving up.
MAX_LOOPS=36          # 36 x 5min = 3 hours
INTERVAL=300
found=0
for i in $(seq 1 "$MAX_LOOPS"); do
  ts=$(date -u +%H:%M:%S)
  scan=$(for from in $(seq 0 250 9500); do
      curl -s --max-time 20 "https://registry.npmjs.org/-/v1/search?text=keywords:pi-package&size=250&from=$from"
    done | grep -c "pi-context-shake")
  if [ "$scan" -gt 0 ]; then
    echo "[$ts] LOOP $i: FOUND in npm search index ($scan hits)"
    found=1
    break
  fi
  echo "[$ts] LOOP $i: not indexed yet (0/39 pages hit)"
  sleep "$INTERVAL"
done
exit $((1 - found))
