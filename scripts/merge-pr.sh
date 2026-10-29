#!/bin/sh
# merge-pr.sh PR [INTERVAL]: merge PR into master through its merge queue, and return only once it has merged or will
# not. `gh pr merge` alone returns as soon as the PR is queued (or set to merge once its checks pass), and the queue can
# still drop it when `test` fails on master with it merged; so this asks GitHub every INTERVAL seconds, 30 unless given,
# which only the tests do. Exits 0 when PR merged, 1 when it did not, saying why, and 2 when gh fails, in gh's words.
# Waits with no deadline of its own: the queue's check timeout drops an entry whose `test` never reports.
# Run again on a PR already queued or set to merge, it waits on that PR's merge, so a wait that exits 2 is re-run.
set -eu
usage() { echo "merge-pr: usage: merge-pr.sh PR [INTERVAL], each a whole number (given: $*)" >&2; exit 2; }
[ $# -eq 1 ] || [ $# -eq 2 ] || usage "$@"
pr=$1
interval=${2-30}
# [LAW:parse-dont-validate] PR is GraphQL's Int! below and INTERVAL is sleep's; neither is found wrong mid-wait.
for n in "$pr" "$interval"; do case $n in '' | *[!0-9]*) usage "$@" ;; esac; done

# [LAW:no-silent-failure] one word for where the PR is, and why for a PR that will not merge. An open PR set to merge
# once its checks pass never reaches the queue when a required check fails or it conflicts with master, and GitHub
# keeps the auto-merge of each, so the queue's timeout never ends the wait; this does. A removal from the queue after
# the auto-merge was enabled is the same: the auto-merge is spent.
# shellcheck disable=SC2016 # $owner, $name and $number are GraphQL's variables, not the shell's.
query='query($owner: String!, $name: String!, $number: Int!) { repository(owner: $owner, name: $name) {
  pullRequest(number: $number) { state mergeable isInMergeQueue mergeQueueEntry { position } autoMergeRequest { enabledAt }
    commits(last: 1) { nodes { commit { statusCheckRollup { contexts(first: 100) { nodes {
      ... on CheckRun { name conclusion databaseId isRequired(pullRequestNumber: $number) }
      ... on StatusContext { context state isRequired(pullRequestNumber: $number) } } } } } } }
    timelineItems(last: 1, itemTypes: [REMOVED_FROM_MERGE_QUEUE_EVENT]) {
      nodes { ... on RemovedFromMergeQueueEvent { createdAt reason } } } } } }'
# [LAW:one-source-of-truth] a check's result is its latest run's, as the gate reads it: a run that a later one on the
# same commit replaced is history. A status context is one per context already.
# shellcheck disable=SC2016 # $failed and $removed are jq's variables, not the shell's.
where='.data.repository.pullRequest
  | ([.commits.nodes[0].commit.statusCheckRollup.contexts.nodes[]? | select(.isRequired)]
    | group_by(.name // .context) | map(max_by(.databaseId)
      | select((.conclusion // .state) | IN("FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE", "STALE"))
      | .name // .context)) as $failed
  | .timelineItems.nodes[0] as $removed
  | if .state != "OPEN" then .state
    elif .isInMergeQueue then "QUEUED at position \(.mergeQueueEntry.position)"
    elif .autoMergeRequest == null then "NOT QUEUED: \(if $removed == null then "never in the merge queue"
      else "removed from the merge queue: \($removed.reason)" end)"
    elif $failed != [] then "WILL NOT MERGE: its required \($failed | join(", ")) failed"
    elif .mergeable == "CONFLICTING" then "WILL NOT MERGE: it conflicts with master"
    elif $removed != null and $removed.createdAt > .autoMergeRequest.enabledAt
      then "WILL NOT MERGE: removed from the merge queue: \($removed.reason)"
    else "WAITING for its checks before it is queued" end'
ask() { gh api graphql -F owner='{owner}' -F name='{repo}' -F number="$pr" -f query="$query" --jq "$where"; }

# Only a PR nothing is merging is merged; one already queued or set to merge is waited on as it is.
now=$(ask) || exit 2
case $now in NOT\ QUEUED*) gh pr merge "$pr" || exit 2; now=$(ask) || exit 2 ;; esac
last=
while :; do
  [ "$now" = "$last" ] || echo "merge-pr: #$pr $now" >&2
  last=$now
  case $now in
    MERGED) exit 0 ;;
    QUEUED*|WAITING*) sleep "$interval" ;;
    # GitHub keeps the auto-merge of a PR that will not merge, so a later push or rerun could merge what this said did not.
    WILL\ NOT\ MERGE*) gh pr merge --disable-auto "$pr" || exit 2; exit 1 ;;
    *) exit 1 ;;
  esac
  now=$(ask) || exit 2
done
