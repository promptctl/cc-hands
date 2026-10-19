#!/bin/sh
# not-rewound.sh COMMIT: refuse COMMIT if it is built on the line master was rewound away from on 2026-10-15.
# That day master was force-pushed from 221b272 back to 50557d9, dropping 25 commits, #270 .. #296, that form one line
# with no merges and begin at FIRST, whose parent is 50557d9; so COMMIT is on the line exactly when FIRST is its
# ancestor. Exits 0 when COMMIT is off the line, 1 when it is on it, and 2 when this clone cannot tell.
set -eu
FIRST=ab3ad395640441644c48b37dfea8884330f1ede7
[ $# -eq 1 ] || { echo "not-rewound: usage: not-rewound.sh COMMIT (given: $*)" >&2; exit 2; }
# [LAW:no-silent-failure] a shallow clone lacks the history between COMMIT and FIRST, so the ancestry test would answer
# "not an ancestor" for a commit that is on the line.
[ "$(git rev-parse --is-shallow-repository)" = false ] || { echo "not-rewound: this clone is shallow, so it cannot tell whether $1 is on the rewound line; check out with the whole history" >&2; exit 2; }
commit=$(git rev-parse --verify --quiet "$1^{commit}") || { echo "not-rewound: $1 is not a commit in this clone" >&2; exit 2; }
# No branch need hold FIRST once the rewound line's branches are gone, so a whole clone may still lack it.
git cat-file -e "$FIRST^{commit}" 2>/dev/null || git fetch --quiet --no-tags origin "$FIRST" || { echo "not-rewound: $FIRST, the rewound line's first commit, is not in this clone and could not be fetched from origin" >&2; exit 2; }
set +e
git merge-base --is-ancestor "$FIRST" "$commit"
code=$?
set -e
case $code in
  0) echo "not-rewound: $1 ($commit) is built on the line master was rewound away from on 2026-10-15, which begins at $FIRST; nothing may go on top of those commits, so the work must be rebuilt on master" >&2; exit 1 ;;
  1) echo "not-rewound: $1 ($commit) is not built on the line master was rewound away from on 2026-10-15" ;;
  *) echo "not-rewound: git could not tell whether $FIRST is an ancestor of $1 ($commit)" >&2; exit 2 ;;
esac
