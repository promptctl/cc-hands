#!/bin/sh
# not-rewound.sh COMMIT [FIRST]: refuse COMMIT if it is built on the line master was rewound away from on 2026-10-15.
# That day master was force-pushed from 221b272 back to 50557d9, dropping 25 commits, #270 .. #296, that form one line
# with no merges and begin at FIRST, whose parent is 50557d9; so COMMIT is on the line exactly when FIRST is its
# ancestor. FIRST is that commit unless given, which only the tests do. Exits 0 when COMMIT is off the line, 1 when it
# is on it, and 2 when this clone cannot tell. Every commit on the line is an ancestor of its last, 221b272, so work
# rebased onto master with 221b272 as its upstream keeps its own commits alone, wherever on the line it began.
set -eu
[ $# -eq 1 ] || [ $# -eq 2 ] || { echo "not-rewound: usage: not-rewound.sh COMMIT [FIRST] (given: $*)" >&2; exit 2; }
# An empty FIRST is not a default; git fails to look it up below, which stays loud.
FIRST=${2-ab3ad395640441644c48b37dfea8884330f1ede7}
# [LAW:no-silent-failure] a shallow clone lacks the history between COMMIT and FIRST, so it cannot tell a commit on the
# line from one off it; a git that cannot answer stops the script here, after its own error.
shallow=$(git rev-parse --is-shallow-repository) || exit 2
case $shallow in
  false) ;;
  true) echo "not-rewound: this clone is shallow, so it cannot tell whether $1 is on the rewound line; check out with the whole history" >&2; exit 2 ;;
  *) echo "not-rewound: git rev-parse --is-shallow-repository answered '$shallow', not true or false" >&2; exit 2 ;;
esac
commit=$(git rev-parse --verify --quiet "$1^{commit}") || { echo "not-rewound: $1 is not a commit in this clone" >&2; exit 2; }
# A whole clone holds every ancestor of COMMIT, so one built on FIRST brings FIRST with it. Once no ref on origin holds
# the line, a clone lacks FIRST, and that absence is the answer: COMMIT is off the line. cat-file -e exits 1 for an
# object that is absent and 128 for any other trouble, which stays loud.
code=0
git cat-file -e "$FIRST" || code=$?
case $code in
  0) ;;
  1) echo "not-rewound: $1 ($commit) is not built on the line master was rewound away from on 2026-10-15; this whole clone does not hold its first commit $FIRST"; exit 0 ;;
  *) echo "not-rewound: git could not tell whether this clone holds $FIRST" >&2; exit 2 ;;
esac
code=0
git merge-base --is-ancestor "$FIRST" "$commit" || code=$?
case $code in
  0) echo "not-rewound: $1 ($commit) is built on the line master was rewound away from on 2026-10-15, which begins at $FIRST; nothing may go on top of those commits, so the work must be rebuilt on master from its own commits alone, with git rebase --onto origin/master 221b272e912309e8750df6f62c6a85a0e71de9fc, the line's last commit; a plain rebase onto master, or one whose upstream is an earlier commit of the line, would bring rewound commits back with it" >&2; exit 1 ;;
  1) echo "not-rewound: $1 ($commit) is not built on the line master was rewound away from on 2026-10-15" ;;
  *) echo "not-rewound: git could not tell whether $FIRST is an ancestor of $1 ($commit)" >&2; exit 2 ;;
esac
