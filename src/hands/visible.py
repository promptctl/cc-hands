"""Text hands shows a person with every character that would act on their terminal written as its escape."""

# Every C0 and C1 control, DEL, the line and paragraph separators, and bidi embedding, override, and isolate, written as
# its JSON escape: what hands shows comes from transcripts, replies, session names, and the audit log, and a raw ESC,
# BEL, or BS in it would move the cursor, ring the bell, or rewrite the line, a raw RLO would show it reversed, a raw
# U+2028 is a line's end to some readers, and a NUL is no argument the OS can hand a program. A line break and a tab are
# layout, and pass.
VISIBLE = {
    code: f"\\u{code:04x}"
    for code in (*range(0x20), *range(0x7F, 0xA0), *range(0x2028, 0x202F), *range(0x2066, 0x206A))
    if chr(code) not in "\n\t"
}
