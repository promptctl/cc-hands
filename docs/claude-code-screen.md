# Claude Code's screen, and when a key reaches its prompt input

Release examined: Claude Code 2.1.308, the release `src/hands/brain/pinned.py` pins.

hands types into the brain's terminal. Return in a dialog answers it, so hands types nothing unless a key would reach Claude Code's prompt input. This document sets out, from Claude Code's own code, everything Claude Code draws that takes keys, what each surface draws, and how hands can tell from `tmux capture-pane` text whether a key reaches the prompt input. It ends with a rule and every case where the rule is wrong.

## How to read this document

Everything here was read from the bundled JavaScript of that release's binary, in memory, through `readGraphFromFile` in `~/code/cc-extra/src/bun-graph.js`. Nothing was run, and nobody logged in. The code is minified React on a fork of Ink, so names are mangled and change between releases. Each fact names the module (chunk) it was read from and a search string. The search string is a JavaScript regular expression, written to match the source text as it is, and the script at the end of this section finds it: `node grep.mjs <binary> '<regex>'` prints every hit with the text around it. The source keeps non-ASCII text as escapes: `…` on screen is the six characters `\u2026` in the source, matched by `\\u2026`. Where this document quotes on-screen text, it shows the characters the terminal draws.

```js
// grep.mjs: search the binary's JS modules in memory. node grep.mjs <binary> <regex> [context] [max]
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const g = require(process.env.HOME + '/code/cc-extra/src/bun-graph.js');
const [bin, pat, ctx = '300', max = '20'] = process.argv.slice(2);
const r = g.readGraphFromFile(bin);
if (!r.ok) { console.error(r); process.exit(1); }
const re = new RegExp(pat, 'g');
let n = 0;
for (const m of r.modules) {
  if (m.loader !== 'js' && m.loader !== 1) continue;
  const t = m.text();
  for (const hit of t.matchAll(re)) {
    if (n++ >= +max) process.exit(0);
    const i = hit.index;
    console.log(`=== ${m.name} @${i}\n${t.slice(Math.max(0, i - +ctx), i + +ctx)}\n`);
  }
}
```

## The brain as hands launches it

What the brain can reach follows from how `src/hands/brain/process.py` starts it:

- **Flags:** `--model`, `--session-id` or `--resume <id>`, `--setting-sources user`, `--system-prompt-snapshot off`, `--mcp-config` with hands' server, `--append-system-prompt`, `--allowedTools mcp__hands` plus hands' plugin skills, `--plugin-dir`, and `--settings` carrying hands' HTTP hooks (UserPromptSubmit, Stop, StopFailure, PermissionRequest, Elicitation), `ENABLE_TOOL_SEARCH=false`, and the `permissions.defaultMode` the brain's own settings.json names.
- **Environment:** `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`, `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, `CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION=false`, `CLAUDE_CODE_DISABLE_CRON=1`, `ENABLE_CLAUDEAI_MCP_SERVERS=false`, `DISABLE_AUTOUPDATER=1`, `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`, `CLAUDE_CONFIG_DIR` set to the brain's home, `ANTHROPIC_BASE_URL` set to hands' proxy. No API key or provider switch is passed on. The rest of hands' environment is passed through.
- **Place:** an empty working directory under the brain's home, in a tmux pane of 200 columns by 50 rows, under fritter.
- **Keys hands sends:** a prompt (pasted in one bracketed paste, then Enter), a slash command (its name typed as keys, its arguments pasted, then Enter), and the keys Escape, Enter, Ctrl-C, Ctrl-U, Up, Down, Tab and Shift-Tab (`src/hands/sessions/tmux.py`).

**Feature gates in the brain.** Claude Code reads its GrowthBook gates through one call that returns the code's default when GrowthBook is off (chunk-dxcygj28.js, search `getFeatureValueWithSource\(e,n\)\{let r=this\.getEnvironmentOverrides\(\)`). GrowthBook is on only when `Bg()` holds (chunk-dxcygj28.js, search `isEnabled:\(\)=>Bg\(\)`, `function Bg\(\)\{return!a\.DISABLE_GROWTHBOOK&&F2\(\)\}`). `F2()` is false while telemetry is restricted (chunk-dxcygj28.js, search `function F2\(\)\{return!n_\(\)\}`; chunk-waz2w8jn.js, search `function n_\(\)\{return LU\(\)\|\|`), and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` restricts it (chunk-2y4p7633.js, search `function W6t\(\)\{if\(process\.env\.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC\)return"essential-traffic"`). So in the brain every `k("tengu_…", default)` gate returns its default. A surface whose only way to open is a gate that defaults off cannot appear in the brain.

**Settings the brain's home decides.** The permission mode, the `tui` setting (fullscreen or not), `editorMode`, a sandbox, extra MCP servers and the brain's own keybindings (a `keybindings.json` in its config directory loads over the defaults: chunk-vkjeenf8.js, search `KeybindingSetup initialized with`) are the brain home's to say. The brain home on the machine this spike ran on names `defaultMode: "auto"`, `skipDangerousModePermissionPrompt: true` and `tui: "fullscreen"`. Those are that home's values, not hands' launch.

## How a key travels

1. **Before the first screen.** At process start Claude Code puts stdin in raw mode and buffers what is typed (chunk-fgyxav5k.js, search `process\.stdin\.on\("readable",e\.readableHandler\)`). A carriage return is kept in the buffer as a newline, not a send; Ctrl-C exits Claude Code with status 130 and Ctrl-D ends the buffering (chunk-fgyxav5k.js, search `if\(o===3\)\{b2\(\),process\.exit\(130\)`). The buffering stops the moment Ink first turns on raw mode (chunk-e4p6x208.js, search `if\(this\.rawModeEnabledCount===0\)\{if\(b2\(\)`), which the first screen that reads keys does, a startup screen or the REPL. When the REPL mounts, what was buffered becomes the prompt's draft, unsent (chunk-3ghxrna3.js, search `ZFe\(Jd,ABt\(\)\)`). So text typed before the first screen lands in the draft unsent, and keys typed while a startup screen is up go to that screen ("Startup screens").
2. **Reading.** Ink's app reads stdin on "readable" (chunk-e4p6x208.js, search `u\.addListener\("readable",this\.handleReadable\)`), parses keys, bracketed paste and mouse. A bracketed paste becomes one paste event for the focused element (chunk-e4p6x208.js, search `if\(D\.kind==="paste"\)\{n\.props\.dispatchPasteEvent\(D\.text\);continue\}`). hands' prompt is pasted, so it arrives as one paste event, never as key presses; the Enter after it is a separate key.
3. **Root capture.** The whole app sits in one box that captures every key first (chunk-vkjeenf8.js, search `keybindingScope:"Global",tabIndex:-1,flexDirection:"column",flexGrow:1,onKeyDownCapture:W`). The key is resolved against the keybinding contexts. A matching action runs the handler that registered it and the key goes no further. The default bindings are one table (chunk-dg7g57st.js, search `TB=\[{context:"Global",bindings:{"ctrl\+c":"app:interrupt"`): Return is `chat:submit` in the Chat context, `confirm:yes` in Confirmation, `select:accept` in Select, `historySearch:execute` in HistorySearch, `footer:openSelected` in Footer; Escape is `chat:cancel`, `confirm:no`, `select:cancel`; Ctrl-C is `app:interrupt` everywhere; Ctrl+B is `task:background` in the Task context only.
4. **Registered handlers.** A component registers handlers through three hooks (chunk-cpb453mc.js, search `function at\(e,o,r=\{\}\)\{let\{context:t="Global",isActive:s=!0\}=r`, `function on\(e,o=\{\}\)\{let\{context:r="Global",isActive:t=!0\}=o`, `function Kf\(e,\{isActive:o=!0\}=\{\}\)`). A handler is live while its component is mounted and active, whatever has focus. Which components are mounted decides where a Return goes.
5. **Focus.** A key no handler took goes to the focused element's `onKeyDown`. Focus is set by `autoFocus`, by the focus hook that takes focus back whenever it moves away (chunk-prsva4j5.js, search `function m_\(e,r,n=!1\)`), and by dialog boxes that claim it. A printable key on a focused text field goes straight to the field.

The prompt input sends a prompt only through its `chat:submit` handler, and it registers that handler only while no overlay is active and history search is closed (chunk-3ghxrna3.js, search `if\(!mg\|\|Wo\|\|bu\){return}let kxt=AOo\(sne\("enter"\),"Chat",mg.bindings\)`). Every other Return binding belongs to one of the surfaces below.

## The prompt input

### How it is drawn

The input row starts with the pointer `❯` (U+276F) and a no-break space, U+00A0 (chunk-3ghxrna3.js, search `children:\[te.pointer,"\\xA0"\]`; the pointer is chunk-xqpq2qwz.js, search `pointer:"\\u276F"`). In bash mode, a draft starting with `!`, it starts with `!` and U+00A0 (chunk-3ghxrna3.js, search `children:"!\\xA0"`). The fullscreen layout builds the same prefix (chunk-3ghxrna3.js, search `te.pointer\)\+knr`; chunk-j6kzk6ef.js, search `knr="\\xA0"`). When the user views another agent, the prefix is that agent's name in place of `❯` (the same expression in chunk-3ghxrna3.js, search `return\(h==="bash"\?"!":x\?pt\(tL\(x.text\)`).

The row sits in a box with a rounded border drawn on top and bottom only, so it shows as a full-width row of `─`, the input row or rows, and another row of `─` (chunk-3ghxrna3.js, search `borderStyle:"round",borderLeft:!1,borderRight:!1,borderBottom:!0`; the fullscreen layout draws both rules with `─`, chunk-j6kzk6ef.js, search `h4="\\u2500"`). The top rule can carry a label, such as an IDE selection. Under the box, dim footer rows carry hints such as `? for shortcuts` (chunk-3ghxrna3.js, search `"\? for shortcuts"`), `Press Ctrl-C again to exit` while an exit is pending (chunk-3ghxrna3.js, search `children:\["Press ",jn," again to"," ",Mr\]`), `-- INSERT --` or `-- NORMAL --` in vim mode (chunk-3ghxrna3.js, search `children:\["-- ",Se," --"\]`), and `Pasting…` while a paste is read.

When a blocking dialog is held back because the user is typing, the line `Waiting for permission…` is drawn above the input (chunk-3ghxrna3.js, search `children:"Waiting for permission`; its condition is `hasSuppressedDialogs`, chunk-3ghxrna3.js, search `function cio\(AKo\){return AKo.hasSuppressedDialogs}`).

### What marks it and nothing else

Of everything Claude Code's own code draws, only the prompt input puts U+00A0 after `❯` or `!`. Every U+00A0 in the graph was read: 42 sites in 19 chunks, written as `\xA0`, `\u00A0` or the character itself. The only ones joined to `❯` or `!` are the prompt's two prefixes above and `knr`, the U+00A0 that `BT`, the fullscreen prefix builder, puts after `❯`, `!` or an agent's name (chunk-3ghxrna3.js, search `function BT\(\{mode:h,agent:x,maxWidth:L\}\)`). The rest sit in regular expressions, entity tables, the transcript's `⎿` and `⏺` rows, and the markdown renderer, below. Every other `❯` the code draws is followed by an ordinary space:

- a past user prompt in the transcript draws `❯`, a space, then the text (chunk-e5ckmb4a.js, search `"aria-label":"you:",color:"inactive",children:te\.pointer`, `children:\[W," ",e\(n,{color:"text",children:B}\)\]`);
- the cursor on a select's focused option is its own box beside the label with a one-cell gap (chunk-amjw57qh.js, search `r\(s,{flexDirection:"row",gap:1,children:\[Ve,ne,E\]}\)`).

Text the model writes is another matter. The markdown renderer turns a space before a list number into U+00A0 (chunk-tcb99wbx.js, search `function ve\(e\)\{return e\.replace\(/ \(\\d\{1,9\}\[\.\)\]\)\(\?!\\w\)/g,"\\xA0\$1"\)\}`), so a line of the brain's reply reading `❯ 1. Yes`, as when it quotes a dialog, is drawn in the transcript as `❯`, U+00A0, `1. Yes`. A row whose first non-blank characters are `❯` and U+00A0, or `!` and U+00A0, is the prompt input's row or a line of transcript text; "The rule" tells them apart only by what is drawn around the row.

### When it is drawn

The REPL draws the prompt input only when `lh` holds (chunk-3ghxrna3.js, search `lh=!Fa&&!Fm&&!Rn`, `visible:lh`):

- `Fa` is `fI()`, true while a user-invoked dialog that hides the prompt is open, or while a blocking dialog is in the dialog store and not held back (chunk-3ghxrna3.js, search `function fI\(\){let L=w\(3\),h=jm\(\),x=gh\(\)!==null`, `return x7\(Z\)\|\|!x&&QE\(Z\)!==null`).
- `Fm` is an exit in progress (chunk-3ghxrna3.js, search `function SKo\(fS\){return fS.exit.kind!=="none"}`).
- In transcript view nothing of the bottom area is drawn (chunk-3ghxrna3.js, search `sm=Fs==="transcript"`).

So while a blocking dialog is drawn, the prompt input is not drawn at all: not dimmed, not covered, absent.

### When it is drawn but does not take keys

The editor takes focus only when `zE` holds (chunk-3ghxrna3.js, search `zE=yp===null&&!bu&&!Wo`), and `Wo` is any active overlay or an open slash-command panel (chunk-3ghxrna3.js, search `Wo=ZY\(\)\|\|Rn`):

- **An overlay in place of the editor** (`yp`): the editor is passed `hidden` and draws nothing, so no prompt row is on screen (chunk-3ghxrna3.js, search `let DDe=yK;if\(Le\){return null}`). See "Overlays drawn in place of the editor".
- **History search** (`bu`): the input rows stay drawn. See "History search".
- **A slash-command panel opened mid-turn**: shown with `hidesPrompt:false`, so the prompt row stays drawn while the panel holds the keys. See "Slash-command panels".
- **A selected footer item**: Down past the end of history selects the first footer item when there is one (chunk-3ghxrna3.js, search `if\(Cd\(fb\),fb==="tasks"&&!ue\(\).hasSeenTasksHint\)`). A focused box then takes keys (chunk-3ghxrna3.js, search `Gl&&!Wo&&e\(s,{tabIndex:0,autoFocus:!0,onKeyDown:Dk}\)`): Return opens the item (`footer:openSelected`, chunk-dg7g57st.js, search `enter:"footer:openSelected"`), and printable keys are passed on into the draft (chunk-3ghxrna3.js, search `Pn.current\?.typeText\(Ik\)`). The selection is drawn as colour only.
- **A focused above-prompt pane** (Ctrl+X Tab, chunk-dg7g57st.js, search `"ctrl\+x tab":"abovePrompt:focus"`): only when a plugin provides panes.

### Keys at the prompt

- **Return:** `chat:submit` (chunk-dg7g57st.js, search `enter:"chat:submit"`). The handler declines while a footer item owns the key (chunk-3ghxrna3.js, search `if\(uu\("chat:submit"\)\){return!1}`). While a turn runs, a submitted prompt is queued, and the prompt stays drawn.
- **Return with the autocomplete list showing:** the editor applies the highlighted suggestion instead of submitting (chunk-3ghxrna3.js, search `if\(\$s==="taken"\){\$n.preventDefault\(\);return}`); a slash-command suggestion is run (chunk-3ghxrna3.js, search `shouldExecute:\$n===void 0`).
- **Return with a one-character draft under an inline digit question:** answers the question with that digit instead of submitting (chunk-3ghxrna3.js, search `function Zm\({inputValue:h,isValidDigit:x,onDigit:L`). See "Above-prompt band, panes, footer selection and surveys".
- **Typed or pasted text:** goes into the draft. In vim NORMAL mode printable keys are editing commands (chunk-3ghxrna3.js, search `function \$f\(\){return K\(\(h\)=>h.settings.editorMode\)`).
- **Escape:** `chat:cancel` stops a running turn. On an empty, idle prompt with history, a second Escape within 800 ms opens Rewind (chunk-3ghxrna3.js, search `of=l5\(Poo,Se.openMessageSelector\)`; the window is chunk-cpb453mc.js, search `var a=800;function l5\(`).
- **Ctrl-C:** `app:interrupt` stops a turn or clears the draft; on an empty prompt the first press shows `Press Ctrl-C again to exit` and a second within 800 ms exits.
- **Ctrl+B:** bound only while a foreground tool's progress is drawn (chunk-59apvq15.js, search `return d.registerHandler\({action:"task:background",context:"Task"`); it moves that tool to the background. At an idle prompt it is unbound.
- **Shift-Tab:** `chat:cycleMode` moves the brain to its next permission mode (chunk-dg7g57st.js, search `d=b\?"shift\+tab":"meta\+m"`, `\.\.\.\{\[d\]:"chat:cycleMode"\}`).
- **Ctrl-U:** deletes the draft from the cursor back to the start of its line (chunk-prsva4j5.js, search `\["u",Tt\]`).

## The dialog store and its host

Every dialog that stops a turn, and every slash-command panel, is an entry in one dialog store (chunk-5a2asxb8.js). An entry is either **blocking** (asked by Claude Code: permissions, questions, notices) or **user-invoked** (a slash-command panel). The host draws the newest blocking entry (chunk-5a2asxb8.js, search `function QE\(o\){return o.open.findLast\(\(e\)=>!e.userInvoked\)\?\?null}`) and the newest user-invoked one (chunk-5a2asxb8.js, search `function gpe\(o\){return o.open.findLast\(\(e\)=>e.userInvoked===!0\)\?\?null}`); a blocking entry wins when both want the same place (chunk-3ghxrna3.js, search `It=St\?Ve:lt\?\$e:null`).

- **Kinds.** Each kind maps to a component in one registry (chunk-3ghxrna3.js, search `var ZRe={...tl\(Oh,mgt\)`) and to its layout, its notification and its status reason in another (chunk-3ghxrna3.js, search `jh={\[Oh.kind\]:{layout:\(h,x\)=>x==="fullscreen"\?"modal"`). Layout is `inline` (in the conversation's flow, just above where the prompt would be), `bottom`, or `modal` (the fullscreen renderer's modal slot). Entries in the registry whose module is compiled out add nothing (chunk-3ghxrna3.js, search `Rft=null,Pft=null,Aft=null`; `SIo=\{\};var bIo=\{\};var wIo=\{\};var kIo=\{\}`).
- **Holding a dialog back.** A blocking dialog is not drawn while `gh()` names a reason (chunk-3ghxrna3.js, search `return h\?"progress":L\?"panel":H\?"draft":x\?"typing":null`, `function hD\(\){let h=eis\(\),x=gh\(\),L=RBe\(\);return x!==null\|\|L\?null:h}`):
  - `typing`: the draft is non-empty and was edited in the last 1500 ms (chunk-3ghxrna3.js, search `this.#b=Ce\?this.#t\(\(\)=>{this.#b=null,OPn\(this.#e,!1\)},CWt\)`, `CWt=1500`); it ends at once when the draft empties;
  - `progress`: a `!` bash-mode command is running (chunk-3ghxrna3.js, search `LBo=\{background_hint:!1,bash_mode_progress:!0,agent_progress:!1\}`);
  - `panel`: a slash-command panel is open and the dialog's `hideWhile` names "panel", which is every kind's default (chunk-2fn15ga5.js, search `var D=\["panel"\];function ls\({hideWhile:e=D`);
  - `draft`: the draft is non-empty and the dialog's `hideWhile` names "draft". The host then draws one dim line in its place, `Claude has a question for you — it shows once you send or clear what you're typing.` or the same with "a suggestion" (chunk-3ghxrna3.js, search `Claude has a question for you \\u2014 it shows once you send`).

  The dialog stays in the store while it is held back. Held back for typing, a `!` command or the draft, it leaves the prompt input drawn and taking keys; held back for a panel, it leaves the keys with the panel ("Slash-command panels").
- **Refusing an answer that came too soon.** The host refuses an answer whose key arrived within 150 ms of the dialog's `shownAt`, within 150 ms of the dialog being displaced by another, or within 150 ms (Escape) or 150 to 600 ms (other keys; 600 when the draft still held text) of the dialog being released from being held back (chunk-3ghxrna3.js, search `if\(ma\(Ce,et.shownAt\?\?0\)\){return!1}`, `return!\(ma\(Ce,ne.displacedAt\)\|\|ma\(Ce,ne.releasedAt,Pt\?_H:ne.releaseGraceMs\)\)`, `graceMs:Be\?IFe:_H`; the constants are chunk-vkp30awb.js, search `var _H=150,to=250,IFe=600`). The time is the moment the key's bytes were read (chunk-e4p6x208.js, search `inputNow=\(\)=>this.readArrivedAt\?\?Date.now\(\)`). `shownAt` is set only when the request asked for it (chunk-3ghxrna3.js, search `shownAt:Ve\?.armInputGrace\?Date.now\(\):void 0`), when another dialog of the same class was already open, when it opens while a slash-command panel is open, when it opens within 150 ms of a blocking dialog closing (chunk-5a2asxb8.js, search `m=t.userInvoked!==!0&&ma\(o,c\)`), when it is revealed from the queue, and again each time its content is updated (chunk-5a2asxb8.js, search `payload:n,shownAt:Date\.now\(\)`). **A first Bash, file or generic ("Tool use") permission dialog of the main conversation has no `shownAt`: a Return that arrives as it is drawn answers it.** Some dialogs also refuse keys for a time after they mount, on their own: "Enter plan mode?" and "Publish a plugin?" among them. A refused answer leaves the dialog up and moves focus back to what was refused (chunk-vgtncd69.js, search `function XE\(r\){let\[t,n\]=y\({attempts:0,refused:null}\)`).
- **Frames.** Permission-style dialogs draw in one frame: a blank row, a full-width top rule of `─` only, a bold title row, then the body (chunk-fe3y8kzn.js, search `borderStyle:"round",borderColor:u,borderLeft:!1,borderRight:!1,borderBottom:!1,marginTop:1`). Most other dialogs and panels draw in the Dialog frame: a rule across the width, a bold title, the body, and a dim footer of key hints joined by ` · `. It binds Escape to cancel and claims focus (chunk-tbnp4j48.js, search `scope:"Confirmation",claimFocus:!0`), and it takes Ctrl-C or Ctrl-D twice within 800 ms to cancel, showing `Press Ctrl-C again to cancel` after the first (chunk-tbnp4j48.js, search `children:\["Press ",F.keyName," again to ",C\?"exit":"cancel"\]`).
- **Choices.** The shared select draws one option per row, `N.` before each label unless it is drawn with `hideIndexes`, and `❯` and a space beside the focused one (chunk-amjw57qh.js). "Enter plan mode?" and many yes/no confirms hide the numbers (chunk-3ghxrna3.js, search `hideIndexes:!0,confirmLabel:St!==null\?St\.node:"Yes, enter plan mode"`). Its keys are the Select context (chunk-dg7g57st.js, search `\{context:"Select",bindings:\{up:"select:previous",down:"select:next",j:"select:next",k:"select:previous"`): Return chooses the focused option (chunk-amjw57qh.js, search `f\["select:accept"\]=\(\)=>{if\(o===!0\)return`), a digit 1 to 9 chooses that option at once (chunk-amjw57qh.js, search `if\(o!=="numeric"&&/\^\[0-9\]\$/.test\(M\)\)`), j, k, the arrows, Ctrl-N, Ctrl-P, Page Up, Page Down, Home and End move, Escape cancels. Other printable keys do nothing unless the focused option is a text field. A paste is dropped unless a text field has focus: the select's box has no paste handler. The yes/no confirm puts "Yes" first and focuses it unless the dialog says otherwise (chunk-2rygk5br.js, search `s=g===void 0\?"Yes":g,i=D===void 0\?"No":D`).

## Permission dialogs

### How a tool's ask becomes a dialog

A tool call that needs permission is routed by one table to its dialog kind; the first match wins (chunk-681adkg3.js, search `matches:\(e\)=>e===hb,dialog:P9t`): WebFetch, Claude in Chrome tools, AskUserQuestion, EnterPlanMode, the publish-plugin tool, ExitPlanMode, Skill, PowerShell, Monitor and Workflow each have their own. Edit, Write, NotebookEdit, Glob, Grep and Read with a path go to the file dialog (chunk-681adkg3.js, search `function AIn\(e\){switch\(e\){case oh:case tk:case eve:case FM:case \$M:case RE:return!0`). Bash goes to the Bash dialog, or to the file dialog when its command parses as a sed edit (chunk-681adkg3.js, search `dialog:fat,buildDescriptor`). Everything else, MCP tools, WebSearch and Agent among them, gets the generic dialog (chunk-681adkg3.js, search `dialog:eDe,buildDescriptor`).

In one synchronous run Claude Code starts the PermissionRequest hooks and pushes the dialog to the store (chunk-681adkg3.js, search `if\(!_&&r.personOnly!==!0\)\(async\(\)=>{if\(h\(\)\)return;let H=await s.runHooks\(`). The dialog is in the store before the hook's HTTP request leaves. Asks marked person-only (some safety checks, plugin-hook refusals, Claude in Chrome floor blocks) post no PermissionRequest hook at all.

A main-conversation ask is queued in ask order: one asked while another is up waits beneath it (chunk-681adkg3.js, search `place:qe\?"under":go\?void 0:"queue"`). The 150 ms input grace is armed only for teammate, async-subagent and remote asks, person-only asks, and auto mode's denial-limit fallback and safety-check dialogs (chunk-681adkg3.js, search `armInputGrace:L\|\|fo\|\|b.denialLimitFallback!==void 0\|\|b.autoDenyWindow!==void 0`).

The dialog closes when it is answered, when the PermissionRequest hook answers first (the hook's win aborts the dialog and the store drops it: chunk-681adkg3.js, search `onWin\(v\){if\(M.offersBlockOutsideReads===!0`, `function Re\(\){oe\?.abort\(\)`), when the turn is stopped, or when an auto-mode timer denies it. Escape cancels: the tool is denied and the turn stops (chunk-681adkg3.js, search `case"cancelled":{if\(ke\({behavior:"deny",message:"User aborted"}\)`). "No" with no feedback also stops the turn; "No" with typed feedback denies and lets the turn go on (chunk-681adkg3.js, search `function Sgr\({feedback:e,contentBlocks:s,isSubagent:n}\){return!e&&!s\?.length&&!n}`).

### What every permission dialog shares

- **Layout:** the permission frame (a blank row, a top rule of `─`, a bold title), a body, the question, the numbered options with `❯` on the focused one, and a dim footer. The title can end `· from the <name> agent`, `· from a subagent` or `· from a workflow` (chunk-fe3y8kzn.js, search `from the \$\{C\} agent`).
- **Question:** `Do you want to proceed?` unless the dialog passes another (chunk-cynn13e7.js, search `question:c="Do you want to proceed\?"`).
- **Footer:** `Esc to cancel`, then `Tab to amend` when the focused option takes feedback (chunk-cynn13e7.js, search `e\(q,{chord:"escape",action:"cancel"}\),Y,re`, `Re=ze&&e\(q,{chord:"tab",action:"amend"}\)`).
- **Return:** chooses the focused option, which on a new dialog is the first, "Yes", unless the dialog defaults to No.
- **Typed text:** nothing, unless the focused option is a text field. Tab turns the focused Yes or No into one, with the placeholder `tell Claude what to do next` or `tell Claude what to do differently` (chunk-cynn13e7.js, search `var Sr=\{accept:"tell Claude what to do next",reject:"tell Claude what to do differently"\}`; chunk-amjw57qh.js, search `if\(f.key==="tab"\){if\(f.preventDefault\(\),u&&C!==void 0\)u\(C\);return}`). A pasted prompt is dropped (the hands record "a permission dialog swallows pasted text and takes the Enter as Yes" in `docs/claude-code-behavior.md` is this).
- **Escape:** denies and stops the turn.
- **Ctrl-C:** no permission dialog binds it. The REPL's `app:interrupt` stops the turn, which cancels the dialog and denies the tool (chunk-3ghxrna3.js, search `at\("app:interrupt",Mr,{context:"Global",isActive:jn}\)`). The REPL's own Escape handler is off while a dialog is drawn (chunk-3ghxrna3.js, search `Qn=EB\(\)==="visible"`).
- **Ctrl+B:** no dialog binds it. With a dialog up the asked tool is not running; Ctrl+B reaches only a tool already running beside it.
- **Shift-Tab:** answers the file dialog when it offers a session-wide row (see "Files"); no other permission dialog registers it.
- **Up, Down, Ctrl-U:** Up and Down move the focus between options. Ctrl-U is not bound in Confirmation or Select and acts only in a focused text field.
- **Prompt input:** not drawn while the dialog is drawn; drawn while it is held back (none of the permission kinds names "draft").
- **Status file:** `waiting`, reason `permission prompt` unless named below.

### Generic: "Tool use"

- Kind `permission_prompt` (chunk-dxcygj28.js, search `kind:"permission_prompt"`).
- Text: title `Tool use` (chunk-cynn13e7.js, search `mn=r\(\$a,{title:"Tool use"`), the tool and its arguments, `Do you want to proceed?`, options `Yes` (chunk-cynn13e7.js, search `\{label:"Yes",value:"yes",feedbackConfig:\{type:"accept"\}\}`), optionally `Yes, and don't ask again for <tool> commands in <dir>` (chunk-cynn13e7.js, search `value:"yes-dont-ask-again"`), optionally `Yes, and switch to auto mode` (chunk-cynn13e7.js, search `IYr=\{workflow:"Yes, and switch to auto mode"`), and `No` (chunk-cynn13e7.js, search `\{label:"No",value:"no",feedbackConfig:\{type:"reject"\}\}`).
- Brain: reachable for any tool without its own dialog that its settings do not allow: WebSearch, Agent, MCP tools of servers in the brain's own setup.

### Bash: "Bash command"

- Kind `permission_bash` (chunk-681adkg3.js, search `kind:"permission_bash"`).
- Text: title `Bash command`, `Bash command (unsandboxed)` or `Bash command (runs on <host>)` (chunk-3ghxrna3.js, search `"Bash command \(unsandboxed\)":"Bash command"`), the command, its description and reason, `Do you want to proceed?` (chunk-3ghxrna3.js, search `e\(n,{children:"Do you want to proceed\?"}\),Ht&&e\(n,{dimColor:!0`), options `Yes`, optionally an editable `Yes, and don't ask again for` row whose prefix is a text field, optionally an auto-mode row, and `No`.
- Keys: as shared; with the editable row focused, typed text edits the prefix and Return accepts it.
- Brain: reachable. Bash is not pre-allowed; in auto mode it still asks when the classifier or a rule says ask, and as auto mode's timed safety-check dialog.

### Files: "Edit file", "Create file", "Read file" and others

- Kind `permission_file` (chunk-681adkg3.js, search `kind:"permission_file"`), for Edit, Write, NotebookEdit, Read, Glob and Grep with a path.
- Text: title `Edit file` (chunk-681adkg3.js, search `title:"Edit file"`), `Create file`, `Overwrite file` (chunk-681adkg3.js, search `h="Overwrite file"`), `Edit notebook`, `Read file`, or `Read outside the working directories` with the question `Allow this read outside the working directories?` (chunk-681adkg3.js, search `Ve="Read outside the working directories"`, `Xe="Allow this read outside the working directories\?"`). A diff sits between the title and the question, which is `Do you want to <verb> <file>?` (chunk-3ghxrna3.js, search `return r\(n,{children:\["Do you want to ",h.verbPhrase`).
- **Shift-Tab answers it.** This is the one permission dialog that registers `confirm:cycleMode`, Shift-Tab's Confirmation binding (chunk-3ghxrna3.js, search `on\(\{"confirm:cycleMode":sn\},\{context:"Confirmation"\}\)`). It chooses the dialog's session-wide yes row, which a write dialog labels with a bold `(shift+tab)` (chunk-3ghxrna3.js, search `et=DS\("confirm:cycleMode","Confirmation","shift\+tab"\)`). If Tab has turned an option into a text field, Shift-Tab turns the field back off and answers nothing; with no session-wide row, as when a safety check withholds it, Shift-Tab does nothing (`Pt` and `so` there are the feedback field's `handleInputModeToggle`: chunk-3ghxrna3.js, search `handleInputModeToggle:Pt,handleFocus:_t`).
- Brain: reachable. The brain's working directory is empty, so a read of any real path is outside it and asks in default mode, and asks with the outside-reads title in auto mode. The auto-mode outside-reads ask has a kind of its own for the bridge (chunk-681adkg3.js, search `kind:"auto_mode_outside_reads"`); in the terminal it is this dialog.

### WebFetch: "Fetch"

- Kind `permission_webfetch` (chunk-681adkg3.js, search `kind:"permission_webfetch"`).
- Text: title `Fetch` (chunk-3ghxrna3.js, search `r\(\$a,{title:"Fetch"`), `Do you want to allow Claude to fetch this content?` (chunk-3ghxrna3.js, search `"Do you want to allow Claude to fetch this content\?"`), options `Yes`, `Yes, and don't ask again for <domain>`, `No, and tell Claude what to do differently` (chunk-3ghxrna3.js, search `"No, and tell Claude what to do differently "`).
- Brain: reachable.

### Skill: "Use skill …?"

- Kind `permission_skill` (chunk-681adkg3.js, search `kind:"permission_skill"`).
- Text: title `Use skill "<name>"?` or `Use this skill?` (chunk-3ghxrna3.js, search `"Use this skill\?"`), `Claude may use instructions, code, or files from this Skill.` (chunk-3ghxrna3.js, search `"Claude may use instructions, code, or files from this Skill\."`), options `Yes`, `Yes, and don't ask again for <skill> in <dir>`, `No`.
- Brain: hands' own plugin skills are allowed by `--allowedTools` and do not ask. Other skills of the brain's setup can.

### PowerShell: "PowerShell command"

- Kind `permission_powershell` (chunk-681adkg3.js, search `kind:"permission_powershell"`). Title `PowerShell command` or `PowerShell command (unsandboxed)` (chunk-3ghxrna3.js, search `"PowerShell command \(unsandboxed\)":"PowerShell command"`), `Do you want to proceed?`.
- Brain: whether the PowerShell tool is enabled on macOS was not traced.

### Monitor: "Claude needs your permission"

- Kind `permission_monitor` (chunk-681adkg3.js, search `kind:"permission_monitor"`). Title `Claude needs your permission`, or `Open WebSocket` for a WebSocket monitor (chunk-3ghxrna3.js, search `"Open WebSocket "`).
- Brain: whether the Monitor tool is enabled for the brain was not traced.

### Claude in Chrome: "Claude in Chrome wants to …"

- Kind `permission_browser` (chunk-681adkg3.js, search `kind:"permission_browser"`). Title `Claude in Chrome wants to <action>` (chunk-3ghxrna3.js, search `Claude in Chrome wants to \$\{H\}`).
- Brain: only with the Claude in Chrome MCP tools, which the brain's launch does not add.

### Publish a plugin: "Publish a plugin?"

- Kind `permission_publish_plugin` (chunk-681adkg3.js, search `kind:"permission_publish_plugin"`). Title `Publish a plugin?`, option `Yes, publish` (chunk-3ghxrna3.js, search `"Yes, publish"`). A key within 150 ms of the dialog mounting, or of a refused key, is refused (chunk-amjw57qh.js, search `function qa\(t=_H\){let o=qi\(\),i=A\(Date.now\(\)\)`).
- Brain: publishing is off while nonessential traffic is disabled (chunk-2b3fghwe.js, search `Publishing is off while nonessential traffic is disabled`); whether the dialog is still asked before that refusal was not traced.

### Workflow: "Run a dynamic workflow?"

- Kind `permission_workflow` (chunk-f1xj3653.js, search `kind:"permission_workflow"`), drawn with the shared prompt (chunk-ac9nfdnf.js, search `Run a dynamic workflow\?`).
- Brain: whether the Workflow tool is available to the brain was not traced.

### Computer use: "Computer Use wants to control these apps"

- Kind `computer_use_approval` (chunk-crwxj7jz.js, search `kind:"computer_use_approval"`). Text `Computer Use wants to control these apps` (chunk-3ghxrna3.js, search `"Computer Use wants to control these apps"`), `Allow for this session (…)`, `Deny, and tell Claude what to do differently` (chunk-3ghxrna3.js, search `"Deny, and tell Claude what to do differently "`).
- Brain: needs the computer-use MCP server, which the brain's launch does not add.

### Sandbox network access: "Network request outside of sandbox"

- Kind `sandbox_network_access` (chunk-3ghxrna3.js, search `kind:"sandbox_network_access"`). Title `Network request outside of sandbox`, `Do you want to allow this connection?` (chunk-3ghxrna3.js, search `"Network request outside of sandbox"`, `"Do you want to allow this connection\?"`), options `Yes`, `Yes, and don't ask again for <host>`, `No, and tell Claude what to do differently`. Requested with the input grace armed (chunk-3ghxrna3.js, search `this.deps.requestDialog\(Ub,{host:h,port:x},{signal:H.signal,place:"under",armInputGrace:!0`). Status reason `sandbox request`.
- Brain: only when the brain's settings turn the sandbox on.

### A message from another session: "Held message from another session"

- Kind `peer_inbound_approval` (chunk-3ghxrna3.js, search `kind:"peer_inbound_approval"`). Text `Held message from another session` (chunk-3ghxrna3.js, search `"Held message from another session"`), options `Deliver this message to Claude` and `Deny — …`.
- Brain: only with cross-session messaging, which the brain's launch does not set up.

### Device MCP and unattended-serving consent

- Kinds `device_mcp_consent` and `unattended_serving_consent` (chunk-w15t6xk4.js, search `kind:"device_mcp_consent"`, `kind:"unattended_serving_consent"`). The startup form of the device consent reads `Let cloud sessions use this machine's MCP servers?` (chunk-2ak2nqx5.js, search `Let cloud sessions use this machine's MCP servers\?`).
- Brain: no path to either was found from the brain's launch; the gates were not traced to the end.

### Auto mode

When the brain's mode is auto, a classifier answers most asks. A dialog is still drawn:

- when the classifier or a rule says ask: an ordinary dialog, no input grace;
- after repeated classifier denials, as a fallback, with the input grace armed;
- for asks a safety check flags: the tool's ordinary dialog, with the input grace armed, denied by itself after 120 s; after three such timeouts the asks are denied with no dialog (chunk-a1cg2ptw.js, search `uye=\{enabled:!0,showDialog:!0,timeoutMs:120000,maxDialogTimeouts:3\}`; off with `CLAUDE_CODE_DISABLE_DANGEROUS_RM_TIMEOUT`).

## AskUserQuestion

- Kind `permission_ask_user_question` (chunk-681adkg3.js, search `kind:"permission_ask_user_question"`). Status reason `input needed` (chunk-3ghxrna3.js, search `\[Fbe.kind\]:{waitingFor:"input needed",notification:ob}`).
- Text: one tab per question; the question as the title (chunk-3ghxrna3.js, search `WZ,\{title:L\.displayQuestion\.text`); numbered options with descriptions; a free-text "Other" row; `Chat about this` (chunk-3ghxrna3.js, search `"Chat about this"`); a `Review your answers` page with Submit (chunk-3ghxrna3.js, search `"Review your answers"`). Footer hints include `tab to switch questions` (chunk-3ghxrna3.js, search `chord:"tab",action:"switch questions"`). Whether the dialog draws the permission frame's top rule was not traced.
- Keys: Return chooses the focused option and moves to the next question or the review; typed text goes into the free-text row only when it has focus; Tab switches questions; Escape cancels, denying and stopping the turn.
- Brain: reachable. hands answers it through the PermissionRequest hook; the dialog is drawn beside the held hook until then.

## Plan mode: entering, and approving the plan

### "Enter plan mode?"

- Kind `permission_enter_plan_mode` (chunk-681adkg3.js, search `kind:"permission_enter_plan_mode"`). Text `Enter plan mode?`, options `Yes, enter plan mode` and `No, start implementing now` (chunk-3ghxrna3.js, search `"Enter plan mode\?"`, `"Yes, enter plan mode"`, `"No, start implementing now"`). A key within 150 ms of the dialog mounting, or of a refused key, is refused.
- Brain: reachable; EnterPlanMode is a default tool.

### "Ready to code?"

- Kind `permission_exit_plan_mode_v2` (chunk-681adkg3.js, search `kind:"permission_exit_plan_mode_v2"`), layout `modal`.
- Text: `Ready to code?`, `Here is Claude's plan:` (chunk-3ghxrna3.js, search `"Ready to code\?"`, `"Here is Claude's plan:"`), the plan, options such as `Yes, clear context (N% used) and use auto mode` (chunk-3ghxrna3.js, search `Yes, clear context\$\{Ve\} and use auto mode`), keep-context rows, and the text row `No, keep planning` with the placeholder `Tell Claude what to change` (chunk-3ghxrna3.js, search `label:"No, keep planning",value:"no",placeholder:"Tell Claude what to change"`).
- Keys: as shared. With `No, keep planning` focused, typed text goes into its field; Return with no text does nothing, Return with text denies with that feedback.
- Brain: reachable once the brain is in plan mode.

## MCP elicitation

- Kinds `mcp_elicitation` and `mcp_elicitation_waiting` (chunk-ryzjejx1.js, search `kind:"mcp_elicitation"`, `kind:"mcp_elicitation_waiting"`) and `mcp_url_elicitation` (chunk-h3023gqe.js, search `kind:"mcp_url_elicitation"`). Layout `bottom`; status reason `input needed`.
- Text: `MCP server “<name>” requests your input` (chunk-3ghxrna3.js, search `Ti="MCP server `, ` requests your input"`), the form's fields, then ` Accept ` and ` Decline` with `❯` on the focused one (chunk-3ghxrna3.js, search `children:" Accept  "`, `children:" Decline"`). The URL form offers `Open in browser`, `I'm done, continue` and `Continue without waiting`.
- Opens: only when the Elicitation hook gives no answer. The hook runs first, and an answer from it returns with no dialog (chunk-px84zev8.js, search `let r=await this.deps.runElicitationHooks\(o,u\?_d\(a\):a,n\)`, then `let f=await p\(AX,`). Some servers' elicitations also wait while the draft holds text (chunk-px84zev8.js, search `alsoHideWhile:\["draft"\]`).
- Brain: hands declares the Elicitation hook and declines, so no dialog is drawn while the hook answers. What Claude Code draws when hands' hook fails or times out was not traced.

## Other dialogs in the dialog store

Each is a blocking entry: while it is drawn the prompt input is not. The status reason is `dialog open` unless named.

- **Spending notice** `cost_threshold` (chunk-gbkgbjjz.js, search `vdn="cost_threshold"`): `You've spent $5 on the Anthropic API this session.`, one option `Got it, thanks!` (chunk-3ghxrna3.js, search `title:"You've spent \$5 on the Anthropic API this session."`, `Z=\[{value:"ok",label:"Got it, thanks!"}\]`). Opens when the session's cost passes $5 and the login is usage-billed claude.ai or an admin or billing role of an API organization (chunk-4b24vz6w.js, search `function s1r\(\){if\(a.DISABLE_COST_WARNINGS\)return!1`). Return, Escape and Ctrl-C twice all acknowledge. Brain: reachable on such a login.
- **Auto mode billing notice** `auto_mode_server_fallback` (chunk-gbkgbjjz.js, search `kdn="auto_mode_server_fallback"`): `We're changing auto mode to no longer charge for classifier requests in Claude Code.` (chunk-a1cg2ptw.js, search `We're changing auto mode to no longer charge for classifier requests in Claude Code\.`). Layout `bottom`. Opens in auto mode for tiers other than pro, max and team (chunk-a1cg2ptw.js, search `if\(!Z1o\(\)\|\|e==="pro"\|\|e==="max"\|\|e==="team"`), with the input grace armed (chunk-a1cg2ptw.js, search `await e\(k8e,BTr\(n\),{signal:r.signal,armInputGrace:!0,`). Return continues in auto mode (chunk-3ghxrna3.js, search `Ce=\[{action:"confirm:yes",hint:"continue",run:\(\)=>H\("continue"\)}\]`); Escape stops the turn; Ctrl-C stops it at once. Brain: reachable in auto mode on such a tier; whether hands' proxy counts as a gateway host for it was not determined.
- **Managed settings** `managed_settings_security` (chunk-3ghxrna3.js, search `kP=ls\({kind:"managed_settings_security"`): `Managed settings require approval`, options `Yes, I trust these settings` and `No, exit Claude Code` (chunk-t386ndqg.js, search `confirmLabel:"Yes, I trust these settings",cancelLabel:"No, exit Claude Code"`). Return approves; Escape exits Claude Code. Brain: reachable when the login's organization serves such settings, which the code cannot tell.
- **Session paused** `refusal_fallback_prompt` (chunk-4b24vz6w.js, search `W\$=ls\({kind:"refusal_fallback_prompt"`): title `Session paused` (chunk-3ghxrna3.js, search `et="Session paused"`), first option `Switch to <model>` or `Switch to the fallback model` (chunk-3ghxrna3.js, search `"Switch to the fallback model"`), then `Edit prompt and retry…`. Opens when the API refuses a main-conversation request and offers a fallback model. Return switches models. Brain: reachable.
- **Usage limit** `fable_overage_consent_prompt` (chunk-a1cg2ptw.js, search `OIe=ls\({kind:"fable_overage_consent_prompt"`): `You've reached your Fable limit` (chunk-3ghxrna3.js, search `You've reached your Fable limit`) and usage-credit options. Brain: reachable only on a model and account that hit that limit.
- **IDE welcome** `ide_onboarding` (chunk-3ghxrna3.js, search `Kk=ls\({kind:"ide_onboarding"`): `Welcome to Claude Code for <IDE>` (chunk-3ghxrna3.js, search `Welcome to Claude Code for `). Return or Escape dismisses. Brain: depends on whether hands' environment, which the brain inherits, names an IDE terminal.
- **LSP plugin** `lsp_recommendation` (chunk-3ghxrna3.js, search `Gk=ls\({kind:"lsp_recommendation"`): `LSP plugin recommendation`, `Would you like to install this LSP plugin?` (chunk-3ghxrna3.js, search `Would you like to install this LSP plugin\?`). Held back by "draft". Brain: depends on the brain home's marketplace data and the files the brain reads.
- **Plugin hint** `plugin_hint` (chunk-3ghxrna3.js, search `Qk=ls\({kind:"plugin_hint"`): `Plugin recommendation` (chunk-3ghxrna3.js, search `title:"Plugin recommendation"`). Held back by "draft". Brain: depends on a command's output naming a plugin in the marketplace cache.
- **Fullscreen offer** `fullscreen_upsell` (chunk-gbkgbjjz.js, search `kNr="fullscreen_upsell"`): `Try the new fullscreen renderer?` (chunk-3ghxrna3.js, search `Try the new fullscreen renderer\?`), options `Yes, try it` and `Not now`. Never opens when the `tui` setting is set (chunk-3ghxrna3.js, search `if\(gt\(\).tui!==void 0\)return!1;`). Return accepts and restarts the renderer. Brain: not on a home that sets `tui`; the remaining checks on one that does not were not traced.
- **Background this session?** `left_arrow_confirm` (chunk-3ghxrna3.js, search `O_=ls\({kind:"left_arrow_confirm"`): `Background this session?` with `Stay` as the cancel (chunk-3ghxrna3.js, search `title:"Background this session\?"`). Opens on Left at the prompt while tasks run. Return confirms. Brain: hands never sends Left; a person at the pane can.
- **Remote Control** `remote_callout` (chunk-jbwv1psa.js, search `GZ=ls\({kind:"remote_callout"`): `Enable Remote Control` (chunk-3ghxrna3.js, search `label:"Enable Remote Control"`). Opens on `/remote-control`. Brain: only if that command is typed into it.
- **iTerm2 split panes** `it2_setup` (chunk-bd0yefw6.js, search `WRt=ls\({kind:"it2_setup"`): `iTerm2 Split Pane Setup`; status reason `input needed`. Opens when an agent-team teammate needs iTerm2's `it2` (chunk-gf0fegj1.js, search `if\(I.needsIt2Setup&&n.requestDialog\)`). Brain: not expected; the backend choice inside tmux was not traced.
- **Auto mode setup** `auto_mode_setup_review` and `auto_mode_flagged_allow` (chunk-3ghxrna3.js, search `CKt=ls\({kind:"auto_mode_setup_review"`, `TKt=ls\({kind:"auto_mode_flagged_allow"`). Opened by `/auto-mode-setup`'s background scan. Brain: only if that command is typed into it and its gate holds; the gate was not traced.
- **Unreachable in the brain**, because their only way to open is a gate whose default is off, or hands' launch: `autocompact_offer` (chunk-3ghxrna3.js, search `let cn=k\(eot,{}\);if\(cn!==h\)h=cn,x=tot\(cn\)`), `resume_return` (chunk-3ghxrna3.js, search `function MBe\(h,x\){if\(!k\("tengu_gleaming_fair",!1\)`), `cold_resume_quota` (chunk-3ghxrna3.js, search `function zoe\(h,x,L\){let H=zDe\(\).safeParse\(k\(KDe,{}\)\)`), `newer_model_offer` (chunk-3ghxrna3.js, search `function aHe\(\){return NOt\(\).safeParse\(k\("tengu_hidden_volcano",null\)\).data}`), `effort_medium_nudge` (chunk-3ghxrna3.js, search `Ce=uIt\(k\(wB\[H\].flag,{}\)\)`), `goal_proposal` (chunk-4xdg61zf.js, search `function _Hn\(\){return k\("tengu_propose_goal",!1\)}`), `ultraplan_launch` and `ultraplan_choice` (chunk-a1cg2ptw.js, search `function oq\(\){return k\("tengu_ultraplan_config",null\)\?.enabled===!0`), `chrome_install_upsell` and `chrome_install_setup` (chunk-sfxr7gch.js, search `k\("tengu_chrome_install_upsell",!1\)`), and `auto_default_nudge`, `Make auto mode your default permission mode?` (chunk-3ghxrna3.js, search `title:"Make auto mode your default permission mode\?"`). That last one opens only when no project, local, flag or policy settings name a default mode (chunk-aevtzmfn.js, search `\["projectSettings","localSettings","flagSettings","policySettings"\].some`), and the user settings name a mode other than auto (chunk-aevtzmfn.js, search `if\(e&&e!=="auto"&&!t&&QG\(u\)\)return e`). The brain's user settings are its settings.json. When that names no mode, the first test fails; when it names one, hands passes the same mode in `--settings`, which Claude Code reads as flag settings, and the second fails.

## Slash-command panels

A slash command of type `local-jsx` draws a panel as a user-invoked entry of kind `local_jsx` (chunk-exfkw7pd.js, search `var Oh=ls\({kind:"local_jsx"`). The release has 84 such command objects, among them `/config`, `/model`, `/resume`, `/permissions`, `/mcp`, `/memory`, `/hooks`, `/theme`, `/status`, `/usage`, `/plugin`, `/skills`, `/help`, `/export`, `/add-dir`, `/login`, `/tasks`, `/rewind` (by Escape twice) and `/exit`.

- **Opens:** when the command runs. Typed at the prompt, it is shown with `hidesPrompt:true` (chunk-z2ctr57c.js, search `O.show\(f,{commandName:Lo\(r\),immediate:P,hidesPrompt:!0,retireAtTurnBoundary:!0}\)`), so the prompt input is not drawn. A command that runs at once in the middle of a turn is shown with `hidesPrompt:false` (chunk-3ghxrna3.js, search `immediate:!0,hidesPrompt:!1,retireAtTurnBoundary:!0`): the prompt row stays drawn while the panel holds the keys, because an open panel sets `Wo` and the editor neither takes focus nor registers `chat:submit`. A command is immediate when its object says so (chunk-ymymmxee.js, search `function Kve\(e,n,r\){let s=e\?.immediate`).
- **Closes:** when the panel closes itself, or at the turn's end for one shown mid-turn.
- **Layout:** `modal` under the fullscreen renderer, `bottom` when immediate, `inline` otherwise.
- **Text:** each panel's own. Most draw in the Dialog frame (rule, bold title, dim footer of hints).
- **Keys:** each panel's own. Panels on the Settings context read printable letters as actions: j and k move, `/` searches, r, d, w, f and t act (chunk-dg7g57st.js, search `"/":"settings:search",r:"settings:retry",d:"settings:periodDay"`). Return or Space selects, Escape goes back. Text typed into a panel is read as keystrokes, never as a prompt, and its Return picks whatever is focused.
- **Status:** `dialog open` while no response streams (chunk-3ghxrna3.js, search `if\(h.isShowingLocalJSXCommand&&!h.isResponseStreaming&&!h.delegatedActive\)return"dialog open"`).
- **Brain:** reachable whenever hands types one of these commands, and by a person at the pane. Each of the 84 panels' own key maps was not read.

## Overlays drawn in place of the editor

When the prompt input renders an overlay, the editor draws nothing (no prompt row) and the overlay takes the keys (chunk-3ghxrna3.js, search `r\(s,{flexDirection:"column",children:\[yp,J0\]}\)`):

- the model picker, `Select model`, on Meta+P (chunk-3ghxrna3.js, search `children:"Select model"`);
- the fast-mode picker, `Fast mode`, on Meta+O (chunk-3ghxrna3.js, search `children:"Fast mode"`);
- the thinking toggle, `Toggle thinking mode`, on Meta+T, which also draws `Do you want to proceed?` (chunk-3ghxrna3.js, search `children:"Toggle thinking mode"`);
- a local workflow's detail, opened from the footer (chunk-3ghxrna3.js, search `e\(Wot,{workflow:\$m`);
- the background tasks dialog, opened by Return on the footer's tasks item or by `/tasks` (chunk-3ghxrna3.js, search `e\(OKt,{onDone:ne.closeBashesDialog}\)`, `e\(IKt,{onDone:nu,storageV5:nr`);
- the fullscreen history picker, `Search prompts`, on Ctrl+R in fullscreen (chunk-3ghxrna3.js, search `}else if\(YM\(\)&&dd\){`);
- the cloud artifacts list, cloud sessions only (chunk-3ghxrna3.js, search `va\("cloud-artifacts-list"\)`);
- the Remote Control panel, opened from the footer (chunk-3ghxrna3.js, search `va\("bridge-dialog"\)`);
- the external editor's wait line, `Save and close editor to continue...`, on Ctrl+G (chunk-3ghxrna3.js, search `"Save and close editor to continue\.\.\."`);
- the auto-mode environment offer, `Teach auto mode about your environment?` (chunk-3ghxrna3.js, search `title:"Teach auto mode about your environment\?"`); what opens it was not traced.

hands sends none of Meta+P, Meta+O, Meta+T, Ctrl+R or Ctrl+G. The background tasks dialog is reachable by Down then Return at the prompt while tasks run, and by `/tasks`.

## Autocomplete

Typing `/`, or `@` and a path, opens a suggestion list under the input. It is the one overlay that leaves the editor focused and `chat:submit` registered. Its keys are Tab to accept, Escape to dismiss, Up and Down to move (chunk-dg7g57st.js, search `\{context:"Autocomplete",bindings:\{tab:"autocomplete:accept",escape:"autocomplete:dismiss"`; registered at chunk-3ghxrna3.js, search `on\(Fs,{context:"Autocomplete",isActive:Wi&&!Ti}\)`). Return applies the highlighted suggestion and, for a slash command, runs it (see "Keys at the prompt"). The prompt row stays drawn. A prompt hands sends escapes a leading `/`; an `@` inside a prompt can still open the list. A slash command hands sends types its `/name` as keys, which opens the list, so with no arguments its Enter is the list's Return; which suggestion is highlighted when the typed name matches a command exactly was not traced.

## History search

Ctrl+R opens it. In the inline renderer the input rows stay drawn and the footer reads `search prompts:` or `no matching prompt:` (chunk-3ghxrna3.js, search `"search prompts:"`, `"no matching prompt:"`). In fullscreen it is the `Search prompts` overlay instead. Its keys (chunk-dg7g57st.js, search `{context:"HistorySearch",bindings:{"ctrl\+r":"historySearch:next",escape:"historySearch:accept",tab:"historySearch:accept","ctrl\+c":"historySearch:cancel",enter:"historySearch:execute"`): Return runs the matched prompt, typed text extends the query, Escape or Tab puts the match in the draft, Ctrl-C cancels. While it is open, `chat:submit` is not registered. hands never sends Ctrl+R.

## Message selector (Rewind)

Escape twice on an empty, idle prompt, or `/rewind`, opens it as a slash-command panel with `hidesPrompt:true` (chunk-3ghxrna3.js, search `let H=\{commandName:"rewind",immediate:!1,hidesPrompt:!0`). Text: `Rewind` (chunk-3ghxrna3.js, search `var aNt="Rewind"`), or `Nothing to rewind to yet.`. The prompt input is not drawn. hands can reach it: two Escapes within 800 ms at an empty, idle prompt with history.

## Transcript view

Ctrl+O switches the screen to the transcript; neither the prompt input nor the dialog host is drawn (`sm`, above). The footer reads `Showing detailed transcript`, and `dialog waiting` when the store holds a dialog (chunk-3ghxrna3.js, search `"Showing detailed transcript"`, `"dialog waiting"`). Escape, q or Ctrl-C leave it (chunk-dg7g57st.js, search `{context:"Transcript",bindings:{"ctrl\+e":"transcript:toggleShowAll","ctrl\+c":"transcript:exit",escape:"transcript:exit"`). Typed text goes nowhere. hands never sends Ctrl+O.

## Above-prompt band, panes, footer selection and surveys

These keep the prompt row drawn:

- **Footer selection:** see "When it is drawn but does not take keys".
- **Above-prompt band and plugin panes:** entered with Ctrl+X Tab; they exist only when a plugin provides them. hands' plugin provides only skills; the brain home's own plugins could provide panes.
- **Inline digit questions:** a numbered question drawn above the prompt, such as the web-setup offer `1: Yes, run /web-setup` (chunk-3ghxrna3.js, search `Yet=\[{key:"1",label:"Yes, run /web-setup"}`), and the memory-recall and feedback surveys. A draft of exactly one valid digit, sent with Return, answers the question instead of being submitted (`Zm`, above). Which of these the brain can be shown was not traced.
- **Feedback survey:** answered with Ctrl+X 1, 2 or 3 (chunk-dg7g57st.js, search `{context:"FeedbackSurvey",bindings:{"ctrl\+x 1":"feedbackSurvey:bad"`). hands sends no Ctrl+X.

## Startup screens

These run before the REPL mounts, one after another (chunk-njr2st7n.js), so no prompt row is drawn and no hook runs. Each one that reads keys ends the early buffering ("How a key travels"), so keys typed while it is up go to it: Return answers it, and on the workspace trust screen Return exits Claude Code.

- **Onboarding** (theme, terminal setup): unless the config has `hasCompletedOnboarding` (chunk-njr2st7n.js, search `if\(v.hasCompletedOnboarding&&a.CLAUDE_CODE_POWERUP_ONBOARDING!=="banner"`). Text `Choose the text style that looks best with your terminal` (chunk-j84edmt2.js, search `Choose the text style that looks best with your terminal`). Brain: done by `hands login`'s first run, and hands does not start the brain until its `.claude.json` says so (`src/hands/brain/process.py`).
- **Workspace trust:** unless the working directory or an ancestor is trusted in the config (chunk-njr2st7n.js, search `if\(O=Ds\(\),O&&!I3e\(\)\)return null`; chunk-dxcygj28.js, search `if\(e.projects\?.\[n\]\?.hasTrustDialogAccepted\)return!0`). Trust is read from the brain's `.claude.json`, so `--setting-sources user` does not change it. Text `Accessing workspace:`, the directory, `Quick safety check: Is this a project you created or one you trust?` (chunk-k0k7hd82.js, search `Quick safety check: Is this a project you created or one you trust\?`). Options `Yes, I trust this folder` and `No, exit`, **focus on No, exit** (chunk-k0k7hd82.js, search `cancelFirst:!0,focus:"cancel",confirmLabel:"Yes, I trust this folder"`): Return exits Claude Code. Brain: unreachable, since hands does not start the brain until its `.claude.json` trusts its working directory or one above it (`src/hands/brain/process.py`).
- **Project MCP servers:** `New MCP server found in this project:` (chunk-kkqhjwdv.js, search `New MCP server found in this project: `). Brain: its working directory holds no `.mcp.json`; hands' server comes from `--mcp-config`.
- **API key:** `Detected a custom API key in your environment` (chunk-3jxq5p27.js, search `title:"Detected a custom API key in your environment"`), only for `ANTHROPIC_API_KEY` in the environment (chunk-dxcygj28.js, search `function hgn\(\){let e=a.ANTHROPIC_API_KEY;`). Brain: hands strips that variable; a key the brain's settings.json puts in its `env` reaches the environment, and hands does not start the brain until its `.claude.json` holds an answer for that key (`src/hands/brain/process.py`).
- **Bypass permissions warning:** `WARNING: Claude Code running in Bypass Permissions mode` (chunk-34ft200q.js, search `title:"WARNING: Claude Code running in Bypass Permissions mode"`), only in that mode without `skipDangerousModePermissionPrompt` (chunk-njr2st7n.js, search `if\(u!=="bypassPermissions"&&!h\|\|c4\(\)\)return null`).
- **Settings error:** `Settings Error` or `Settings Warning` (chunk-hqrt52d9.js, search `"Settings Error":"Settings Warning"`), when a settings file has invalid entries. Brain: reachable if its settings.json does.
- **Development channels**, **Claude in Chrome onboarding**, **CLAUDE.md external includes**, **Grove policy**, **Pro trial**: `WARNING: Loading development channels` needs a flag hands does not pass (chunk-4byvvgs4.js, search `title:"WARNING: Loading development channels"`); the Chrome screens need Chrome enabled; the CLAUDE.md includes screen is moot under `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`; the Grove and Pro trial screens depend on the account and were not traced (chunk-njr2st7n.js, search `\{GroveDialog:S\}`, `\{ProTrialStartScreen:S\}`).
- **Resume picker:** only for `--resume` with no id; hands always passes one.
- **The rest.** The table under \"Every place that takes keys\" maps more screens here: the login and provider setup wizards, provider model pinning, the model pin update offer, the onboarding tour offer, consumer terms, the configuration error reset, the org-config block, managed settings, the Claude in Chrome extension notice, the remote spawn and remote home settings choices, device MCP consent, reconsent and re-enrolment, the cloud create checklist and the teleport repository check. Each is drawn by the startup sequence or launched by the startup code as a screen of its own (chunk-nzkvdk15.js, search `Bi=oe\(\(\)=>ve\(\)\.launchDeviceMcpConsentDialog\)`), before the REPL mounts, so no prompt row is drawn while it is up. Which of them the brain can be shown was not traced.

## Global keys and the exit states

- Ctrl-C twice within 800 ms on an empty prompt exits Claude Code; Ctrl-D likewise (`app:exit`).
- Ctrl-T, Ctrl-O, Ctrl-R and Ctrl+] toggle the task list, open the transcript, open history search and open the artifacts picker. hands sends none of them.
- Leaving with tasks running asks `Exit and stop tasks` or move them to the background (chunk-3ghxrna3.js, search `Exit and stop tasks`); leaving a worktree asks to keep or remove it (chunk-3ghxrna3.js, search `Keep worktree`). While an exit is in progress the prompt input is not drawn (`Fm`).

## Timing against the hooks and the status file

- **PermissionRequest hook.** The dialog is pushed to the store in the same synchronous run that starts the hook, before its HTTP request leaves ("How a tool's ask becomes a dialog"). The dialog reaches the screen on the next frame. The hooks run through an async generator (chunk-681adkg3.js, search `async runHooks\(h,D,F,T,B,E\)\{if\(n\.forRemoteExecution===!0\)return null;for await\(let x of L1e\(`), and the request then crosses loopback to hands' listener, so the code does not fix whether hands has the hook before that frame reaches the pane. Person-only asks post no hook.
- **Frames.** Claude Code writes a frame at most every 16 ms, every 4 ms for 50 ms after input, and holds a frame back in 250 ms steps while more than 16384 bytes wait to reach stdout (chunk-z44ns0zs.js, search `var Zg=16,Tls=50,Als=4,Rls=16384,yeo=250`; chunk-e4p6x208.js, search `f=n<this.inputPriorityUntil\?Als:Zg`).
- **An answered dialog** leaves the store at once (chunk-5a2asxb8.js, search `answer\(t,n\){if\(e.getState\(\).open.find\(\(a\)=>a.id===t\)\?.countedAsk\?.settle\(\),!r\(t\)\)return`), and so does one whose hook won. It stays drawn until the next frame: about 4 ms after a key answered it, about 16 ms after a hook did, longer while stdout is backed up. A dialog queued beneath it is revealed in the same store update (chunk-5a2asxb8.js, search `shownAt:Date.now\(\),revealSeq:\(r.revealSeq\?\?0\)\+1`), so it is drawn in the same frame the answered one leaves. With none queued, the prompt input is drawn again in that frame.
- **Status file.** The status is `waiting` whenever the store's top blocking entry exists, drawn or held back (chunk-3ghxrna3.js, search `function TEo\(h\){if\(h.queuedElicitation\)return"input needed";if\(h.topDialogWaitingFor!==void 0\)return h.topDialogWaitingFor`, `function XRe\(h,x\){let L=x\?qU\(h\):QE\(h\);return Rpt\(L\?.kind\)}`). The reason is the kind's (chunk-3ghxrna3.js, search `function Rpt\(h\){return h===void 0\?void 0:jh\[h\]\?.waitingFor\?\?"permission prompt"}`): `permission prompt` for permission kinds, `input needed` for AskUserQuestion, elicitations and iTerm2 setup, `sandbox request`, `goal proposal`, `dialog open` for the rest, and `dialog open` while a slash-command panel shows and nothing streams. While a turn runs, the "suggestion" kinds do not count. The file is written by an effect after the render that changed the status (chunk-3ghxrna3.js, search `co=\(\)=>{Yze\({status:Yo,waitingFor:ao`), asynchronously; whether it lands before or after the frame of the same change is not fixed by the code. `waiting` with the prompt input drawn means a dialog held back, or one whose first frame is not yet on screen.
- **Notification hook, `permission_prompt`.** It is not sent when the dialog appears. A drawn or held-back dialog starts a timer; the hook fires at the first 6 s tick of that timer at which there has been no user input for 6 s, so 6 s after the dialog opened if nothing was typed, and up to 12 s after if something was (chunk-3ghxrna3.js, search `function GPe\(h,x,L\)`, `function YNo\(h,x\){return Date.now\(\)-Math.max\(Cf\(\),x\)>=h}`; chunk-6dp77wfw.js, search `Ion=6000`). A dialog answered sooner never fires it. Its message for permission kinds is `Claude needs your permission` (chunk-3ghxrna3.js, search `ob=\{text:"Claude needs your permission"\}`). `CLAUDE_CODE_DISABLE_PERMISSION_PROMPT_NOTIFY_HOOKS` turns off another path, the SDK's, not this one (chunk-z1fhep8d.js, search `if\(a.CLAUDE_CODE_DISABLE_PERMISSION_PROMPT_NOTIFY_HOOKS\)return\(\)=>{}`).
- **Held back and released.** A dialog held back for typing is released 1500 ms after the last edit, or at once when the draft empties, as it does when the prompt is sent. On release the 150 ms grace (600 ms if the draft still holds text) applies.

## What `capture-pane` returns

hands reads the pane with `tmux capture-pane -p -t <pane>` and strips the trailing whitespace of the whole text, not of each row (`src/hands/sessions/tmux.py`). Without `-e` the text carries no colour or attributes; without `-J` a row the terminal wrapped comes back as two lines; without `-S`/`-E` only the visible rows come back, never the pane's history; without `-a` or `-M` it is the grid the pane shows now, whichever screen (main or alternate) is active.

### Fullscreen or inline

Claude Code draws in one of two modes, chosen once at startup by `Bl()` and read through `Wx()` (chunk-55qwcww5.js, search `function Re\(\)\{return Bl\(\)\?"fullscreen":"inline"\}`):

- **fullscreen** draws on the terminal's alternate screen, the whole UI laid out in the pane's rows: transcript in a scroll box, then the dialog or the prompt input and its footer at the bottom. Nothing scrolls into tmux's history.
- **inline** draws on the main screen: finished transcript rows scroll up into tmux's history, and the live part (spinner, dialog, prompt input, footer) is redrawn at the bottom of the screen.

`Bl()` (chunk-9bc9m9x2.js, search `function Bl\(e=yq\)\{if\(yu\(\)==="local-agent"\)return!1`) decides, in this order: the local-agent entrypoint means inline; `CLAUDE_CODE_SESSION_KIND=bg` means fullscreen; screen-reader mode means inline; `CLAUDE_CODE_NO_FLICKER` read as false, or `CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN` set, means inline (chunk-9bc9m9x2.js, search `CLAUDE_CODE_NO_FLICKER===!1\|\|a.CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN`); `CLAUDE_CODE_NO_FLICKER` read as true means fullscreen; fullscreen turned off after repeated crashes means inline; tmux in iTerm2 control mode (`-CC`) means inline; Windows over SSH means inline; then the `tui` setting, `"fullscreen"` or `"default"`, or with no `tui` set, `CLAUDE_CODE_TUI_TRIAL=fullscreen` in the environment (chunk-9bc9m9x2.js, search `this\.mode=a\.CLAUDE_CODE_TUI_TRIAL==="fullscreen"`) (chunk-9bc9m9x2.js, search `switch\(gt\(\)\.tui\?\?kLt\(\)\)\{case"fullscreen":return!0;case"default":return!1\}`); then a fresh install is fullscreen: a global config with `firstStartVersion` set and `fullscreenUpsellSeenCount` under 3, checked only while GrowthBook is off (chunk-9bc9m9x2.js, search `return e\.firstStartVersion!==void 0&&\(e\.fullscreenUpsellSeenCount\?\?0\)<CLt`). `firstStartVersion` is written with `firstStartTime` on a config's first start (chunk-dxcygj28.js, search `firstStartTime:n,firstStartVersion:`). The remaining two inputs are GrowthBook gates (`tengu_amber_creek`, `tengu_pewter_brook`), which return their default, false, in the brain (see "Feature gates in the brain").

So the brain's mode is its config directory's to say: `tui` in its settings.json, or `firstStartVersion` and `fullscreenUpsellSeenCount` in its `.claude.json`, or `CLAUDE_CODE_NO_FLICKER`, `CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN` or `CLAUDE_CODE_TUI_TRIAL` in the environment hands passes on. The brain home on the machine this spike ran on sets `"tui": "fullscreen"`, so that brain draws fullscreen. A rule that holds in both modes reads only the bottom of the pane, where both modes draw the dialog or the prompt input.

### Size, wrapping and redraws

- The brain's pane is made 200 columns by 50 rows (`src/hands/sessions/pseudoterminal.py`). hands' own client attaches with `-f ignore-size`, so it does not size the window; a person who attaches a client without that flag resizes it under tmux's `window-size` option. fritter passes each size change on to Claude Code's terminal (on SIGWINCH, `fritter/wrap.go`), and Claude Code redraws at the new size.
- Claude Code wraps its own text to the width it was given. A row that reaches the last column can still be wrapped by the terminal, and then `capture-pane -p` returns it as two lines: a marker must be matched on one row's text, never across rows.
- A dialog or prompt taller than the screen loses its top rows: in inline mode they scroll into history, in fullscreen mode the layout clips them. The rows nearest the bottom (a dialog's options and footer, the prompt input and its footer) are the ones always on screen, so markers are taken from there.
- Claude Code paces frames: at most one each 16 ms, one each 4 ms for 50 ms after input, and a frame is held back 250 ms while more than 16384 bytes wait to reach stdout (chunk-z44ns0zs.js, search `var Zg=16,Tls=50,Als=4,Rls=16384,yeo=250`; used at chunk-e4p6x208.js, search `f=n<this\.inputPriorityUntil\?Als:Zg`). Each frame's bytes pass through fritter to the pane, and tmux applies them as it reads them. A capture taken while tmux has applied part of a frame returns rows of two frames. Claude Code can mark where each frame begins and ends (synchronized output), but not when `TMUX` is set (chunk-jy8kzjv7.js, search `return r&&a\.TMUX==null&&process\.env\.ZELLIJ==null`), and tmux sets `TMUX` for every process in its panes, the brain's among them. So the brain's frames reach tmux unmarked. While a frame's bytes keep flowing, two captures that agree, taken a frame apart, rule a torn frame out. While stdout is backed up, the rest of a frame can wait in the pipe from Claude Code through fritter to tmux for longer than any fixed gap, and two captures agree on the torn grid.

### What the pane can show that is not Claude Code's

- A pane in a tmux mode (copy mode, a person scrolling back) still captures Claude Code's grid, but `send-keys` keys go to the mode, not to Claude Code. tmux reports this as `#{pane_in_mode}`, not as text.
- A person attached to the brain's session types into the same input. Nothing on screen says a person is typing.

## The rule

Read the pane with `tmux capture-pane -p`, as `src/hands/sessions/tmux.py` does. A key typed now reaches the prompt input when all four hold, and does not otherwise:

1. **A prompt row is on screen.** Some row's first non-blank characters are `❯` (U+276F) followed by U+00A0, or `!` followed by U+00A0. Take the last such row: Claude Code's own drawing puts either pair only on the prompt input, but transcript text above it can show one too ("What marks it and nothing else"). The prompt input is drawn only while no blocking dialog, no prompt-hiding panel, no overlay in place of the editor, no transcript view and no exit is on screen ("When it is drawn").
2. **The row is the input box's first row.** The row directly above it begins with `─`, the box's top rule, which may carry a label.
3. **Nothing below it holds the keys.** No row below it begins with `❯` and an ordinary space, the cursor of a select, numbered or not, or of the autocomplete list; and no row below it reads `search prompts:` or `no matching prompt:`.
4. **The screen is settled.** A second capture, taken at least 20 ms after the first, returns the same text. Claude Code writes a frame at most every 16 ms, so while its output flows, a capture that caught tmux part-way through a frame differs from the next one. While its stdout is backed up, it does not ("Size, wrapping and redraws").

tmux returns U+00A0 as its own character, not as a space: on tmux 3.7b, a pane that printed `❯`, U+00A0 and `hello` captured as the bytes `e2 9d af c2 a0` followed by `hello`. Python's `str.strip()`, `rstrip()` and `split()` treat U+00A0 as whitespace, so a row that captures as `❯` and U+00A0 alone, as an empty prompt's row can, becomes a bare `❯` once stripped. Match a row before stripping it, or strip only ordinary spaces.

## Where the rule is wrong

Every surface in this document, and the verdict of the rule on it:

| surface | the rule says | right? |
|---|---|---|
| Prompt input, idle or with a turn running (Return queues) | reaches | right |
| Prompt input with a dialog held back for typing, a `!` command or the draft (`Waiting for permission…` marks the typing hold, the dim "Claude has a question for you" line the draft hold) | reaches | right: keys reach the prompt; the dialog is drawn once the draft empties, as it does when Return sends it, with the 150 ms grace |
| Every permission dialog, AskUserQuestion, both plan dialogs, MCP elicitation | does not reach | right: no prompt row is drawn while they are |
| Every other dialog in the dialog store | does not reach | right, for the same reason |
| Slash-command panel opened from the prompt, and the message selector | does not reach | right: shown with `hidesPrompt:true` |
| **Slash-command panel opened mid-turn** (an immediate command typed while a turn runs) | reaches | **wrong**: the prompt row stays drawn while the panel holds the keys, and nothing in the text says which has them |
| Overlays in place of the editor (pickers, background tasks dialog, fullscreen history picker, external editor) | does not reach | right: the hidden editor draws no prompt row |
| Autocomplete list | does not reach, by condition 3: the selected suggestion is drawn as `❯` and a space (chunk-57wwsvpp.js, search `g=f&&!k\?"":c\?`) | right for Return, which applies the highlighted suggestion, and runs it for a slash command, instead of sending the draft; wrong, on the safe side, for text. A slash command hands sends with no arguments presses its Enter with the list open |
| History search, inline | does not reach | right, by condition 3 |
| Transcript view | does not reach | right |
| **Footer item selected** (Down past the end of history while footer items show) | reaches | **wrong**: Return opens the item; the selection is drawn as colour only |
| **Above-prompt pane focused** (Ctrl+X Tab, plugin panes only) | reaches | **wrong**: the pane holds the keys |
| **Inline digit question above the prompt, draft of one digit** | reaches | **wrong for Return**: it answers the question instead of sending the digit |
| **Vim NORMAL mode** (only if the brain's settings name vim) | reaches | **wrong for text**: printable keys are editing commands; Return still sends |
| Startup screens | does not reach | right: keys go to the screen, and Return on workspace trust exits Claude Code |
| Exit in progress, exit and worktree questions | does not reach | right |
| Viewing another agent (the prompt's prefix is the agent's name) | does not reach | wrong, on the safe side: hands waits though the prompt takes keys |
| Screen-reader mode (the prefix is `$` and U+00A0: chunk-3ghxrna3.js, search `Ce=L\?e\(Q,\{children:"\$\\xA0"\}\)`) | does not reach | wrong, on the safe side |
| `TERM=linux` (Claude Code's pointer is then `>`: chunk-xqpq2qwz.js, search `if\(a\.platform!=="win32"\)return i!=="linux"`) | does not reach | wrong, on the safe side; tmux sets its own `TERM` in a pane |
| Input taller than the pane, its first row off screen | does not reach | wrong, on the safe side |
| **A dialog pushed to the store but not yet drawn** | reaches | **wrong**: the screen lags the store by up to one frame, 16 ms, longer while stdout is backed up. A hook-backed ask has started its PermissionRequest hook by then, but the code does not fix whether hands has it before the frame; a person-only ask posts none |
| **Keys in flight when a dialog appears** (between the capture and the keys' arrival) | reaches | **wrong**: a first Bash, file or generic main-conversation permission dialog has no input grace, so a Return arriving as it is drawn answers it, as does a Shift-Tab on a file dialog that offers a session-wide row. Asks with the grace armed refuse a key within 150 ms of being pushed to the store, not of being drawn, so a frame held back while stdout is backed up can use the window up |
| **A frame torn while stdout is backed up** | reaches, when the torn grid still holds the prompt row | **wrong**: two captures agree on a grid that is half one frame and half the next, so condition 4 passes while a dialog is being drawn |
| **Pane in a tmux mode, or a person typing at the pane** | reaches | **wrong**: copy mode takes `send-keys` keys, and a person shares the input. tmux reports a mode as `#{pane_in_mode}`, not in the text |
| **Transcript text imitating the prompt** (a row beginning `─`, then a row beginning `❯` and U+00A0, which the markdown renderer draws from an ordinary `❯ 1.` line of the brain's reply) above a dialog whose choices are not select rows, such as an elicitation form | reaches | **wrong**: condition 3 catches dialogs whose choices are select rows, not one whose choices are buttons |

## What the code does not settle

- Whether the brain draws fullscreen: that is its home's `tui` setting, its `.claude.json`, or its environment ("Fullscreen or inline").
- Whether the PowerShell, Monitor, Workflow and publish-plugin tools, and the device MCP and unattended-serving consents, are open to the brain: their gates were not traced to the end.
- Whether AskUserQuestion draws the permission frame's top rule.
- What Claude Code draws when hands' Elicitation hook fails or times out.
- Whether the status file's write or the frame of the same change lands first.
- Whether hands receives a PermissionRequest hook before the dialog's first frame reaches the pane.
- Whether hands' proxy counts as a gateway host for the auto-mode billing notice.
- The account-dependent screens: managed settings, the Grove policy, the Pro trial, the cost notice.
- Each slash-command panel's own key map (84 panels, not each read), which inline digit questions the brain can be shown, and what opens the auto-mode environment offer.
- The order in which two live handlers for the same key, in different contexts, are tried on a surface that keeps the prompt mounted: it follows mount order, and this spike did not prove that no such pair exists on the above-prompt band, the footer, the surveys, the inline digit questions, autocomplete or history search.
- The remaining checks of the fullscreen offer, and whether the IDE welcome, LSP and plugin recommendations can appear: they depend on hands' environment and the brain home's marketplace data.

## Every place that takes keys

This list is how a reviewer checks that no surface is missing. It was built by script over every JavaScript module in the binary, and it covers:

- every call of the keybinding hooks `at`, `on` and `Kf` (223 calls in the 60 chunks that import them from chunk-cpb453mc.js, none under another name), every `registerHandler`, `registerPreDispatch` and `registerActiveContext` call, and every use of the active-context hook (chunk-khk1ge41.js, search `return r.registerActiveContext\(n\),\(\)=>{r.unregisterActiveContext\(n\)}`);
- every overlay registration (chunk-09t8gemt.js, search `function va\(e,t\)`), every focus-hook call, and every scoped-binding box (chunk-d8fzwfwq.js, search `function bE\({scope:n,bindings:l,active:r=!0,preemptive:s=!1,swallowAll:d=!1,claimFocus:a=!1`);
- every `onKeyDown`, `onKeyDownCapture`, `onPaste`, `onPasteCapture`, `onAction`, `onActionCapture`, `keybindingScope`, `autoFocus` and `textEntry` prop;
- every raw stdin listener and every listener on Ink's app events. No component subscribes to raw input the way Ink's `useInput` does; it is absent;
- every render of a shared key-taking component: the Select (chunk-amjw57qh.js, search `function Ye\(t\){let a=w\(12\);if\(it\(\)\)`), the yes/no Confirm (chunk-2rygk5br.js), the permission prompt (chunk-cynn13e7.js, search `question:c="Do you want to proceed\?"`), the Dialog frame (chunk-tbnp4j48.js), the List (chunk-3kb5ba6v.js), the key box (chunk-q43kb6tz.js), the search box (chunk-x2hxr175.js), Tabs (chunk-evaj6163.js), the text inputs (chunk-jxftkk1f.js, chunk-55qwcww5.js), the forms (chunk-mvx77wcz.js, chunk-wxcct5md.js), and the login picker (chunk-k75ham60.js).

That is 1145 registrations in 143 chunks. Below, they are grouped by the component that holds them: 487 rows, each with its chunk, a search string from the component (its on-screen text where it has one, otherwise the head of its function), what it is, its registrations by kind and count, and the section of this document that covers it. "Dialog frame", "Select", "Confirm", "List", "TextInput" and the like name the shared components above; "at hook", "on hook" and "pre-dispatch" are the keybinding hooks; "overlay" is an overlay registration. The keybinding contexts map to sections as follows:

| context | where it is registered | covered by |
|---|---|---|
| Confirmation | Dialog frame, permission prompt, dialogs, panels, startup screens | Permission dialogs; Other dialogs in the dialog store; Slash-command panels; Startup screens |
| Settings | settings panels and setup wizards | Slash-command panels; Startup screens |
| Select | the shared List and Select | the surface that draws the list |
| Global | Ctrl-C, Ctrl-D, Ctrl-T, Ctrl-O, Ctrl-R, Ctrl+], diff panel keys | Global keys and the exit states |
| Chat | the prompt input | The prompt input |
| Autocomplete | the suggestion list | Autocomplete |
| HistorySearch | Ctrl+R search and the fuzzy picker | History search |
| Transcript | the transcript view | Transcript view |
| Task | Ctrl+B while a foreground tool runs | The prompt input (Keys at the prompt) |
| Footer, AbovePrompt, AbovePromptInput, AbovePromptSelect, Pane, PaneField, FeedbackSurvey | the footer, the above-prompt band and panes, surveys | Above-prompt band, panes, footer selection and surveys |
| Tabs | AskUserQuestion, surveys, panels | the surface that draws the tabs |
| Plugin, DiffDialog, ThemePicker, ModelPicker, EffortSlider, Help, ProactivityMenu | /plugin, /diff, /theme, /model, /effort, /help, the proactivity menu | Slash-command panels |
| Scroll, DiffPanel, Attachments | fullscreen scrolling and selection; the diff side panel (Ctrl+X B only); image attachments (arrows, Backspace, Delete, Escape) | none: no Return or text binding |
| Agents | defined in the defaults, registered by nothing in the REPL | none |
| Terminal, fork, info | not keybinding contexts: a VS Code keybinding writer, a skill's context, a command category | none |

| chunk | search | what it is | registrations | covered by |
|---|---|---|---|---|
| chunk-02s6b4r5.js | `Auto-compact window` | slash command /autocompact | on hook×2, Dialog frame | Slash-command panels |
| chunk-0562vqpw.js | `Attach cancelled` | slash command /artifacts | on hook×3, overlay, exit keys | Slash-command panels |
| chunk-0562vqpw.js | `No artifacts yet\. Publish one with the Artifact tool\.` | slash command /artifacts | Select×2, onKeyDown, onPaste, autoFocus, TextInput, List, Tabs, SearchBox | Slash-command panels |
| chunk-0g31eacc.js | `!0,this\._s` | MCP StdioServerTransport (stdin is JSON-RPC, not a TTY) | stdin data | None: not part of the interactive session the brain runs |
| chunk-0jz7jsdg.js | `You are already on the highest Max subscription plan\. For additional u` | login method picker after /upgrade | LoginPicker | Slash-command panels |
| chunk-1km155we.js | `Save the conversation to a file in the current directory` | slash command /export | at hook, Select, TextInput, Dialog frame | Slash-command panels |
| chunk-296d6jaf.js | `Please select the servers you want to import:` | 'claude mcp add-from-desktop' and other mcp CLI handlers (Ink outside the REPL) | Dialog frame, List | None: not part of the interactive session the brain runs |
| chunk-2ak2nqx5.js | `Let cloud sessions use this machine's MCP servers\?` | device MCP consent (launchDeviceMcpConsentDialog) | at hook | Startup screens |
| chunk-2ak2nqx5.js | `Let cloud sessions use this machine's MCP servers\?` | device MCP consent (launchDeviceMcpConsentDialog) | Select, Dialog frame | Startup screens |
| chunk-2rygk5br.js | `function lr\(L\)\{let a=w\(3` | shared confirm lr/Jqe/Ucr definitions; each use listed | Select, YesNo | How a key travels (shared component: each use is listed under its own row) |
| chunk-2t6p6ynd.js | `function os\(i,r,e=!0\)\{le` | os/VTt: app:interrupt + app:exit (Ctrl-C / Ctrl-D twice) used by dialogs; each use listed | on hook | Global keys and the exit states |
| chunk-2ty33rg3.js | `Run ultrareview in the cloud\?` | slash command /ultrareview | overlay, Dialog frame | Slash-command panels |
| chunk-2ty33rg3.js | `More information: ` | slash command /ultrareview | Select×2 | Slash-command panels |
| chunk-34ft200q.js | `By proceeding, you accept all responsibility for actions taken while r` | Bypass Permissions mode warning | Confirm, Dialog frame | Startup screens |
| chunk-3587hx00.js | `Use this machine's settings in cloud sessions\?` | remote home settings (launchRemoteHomeSettingsDialog; also /config) | at hook | Startup screens |
| chunk-3587hx00.js | `Use this machine's settings in cloud sessions\?` | remote home settings (launchRemoteHomeSettingsDialog; also /config) | Select, Dialog frame | Startup screens |
| chunk-39ybe06s.js | `function Je\(r\)\{return so` | Claude in Chrome MCP server process | stdin listener×2 | None: not part of the interactive session the brain runs |
| chunk-3ghxrna3.js | `function ene\(\{onScreen:h` | pre-dispatch for cold_resume_quota (trigger idle): printable keys go to the prompt draft while the dialog is drawn; Return swallowed after typing/paste until an arrow key | pre-dispatch | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Background this session\?` | left_arrow_confirm ("Background this session?", Stay) | Confirm, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `\(worktree handed off\)` | left_arrow_confirm variant ("Background this session?") | Confirm, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `function Pt\(\)\{let Ht=Se\.` | Scroll context (pageup/pagedown/wheel) in fullscreen; no Enter/printable binding | on hook | None: takes no Return or text in place of the prompt input |
| chunk-3ghxrna3.js | `Scroll wheel is sending arrow keys \\xB7 use PgUp/PgDn to scroll` | Scroll/Transcript scroll and selection copy bindings; no Enter/printable binding | on hook×2 | None: takes no Return or text in place of the prompt input |
| chunk-3ghxrna3.js | `Scroll wheel is sending arrow keys \\xB7 use PgUp/PgDn to scroll` | selection extend (shift+arrows) in fullscreen | on hook, pre-dispatch | None: takes no Return or text in place of the prompt input |
| chunk-3ghxrna3.js | `Scroll wheel is sending arrow keys \\xB7 use PgUp/PgDn to scroll` | arrow-burst detector (scroll wheel sending arrows) notice | app event listener | None: takes no Return or text in place of the prompt input |
| chunk-3ghxrna3.js | `Showing detailed transcript` | Global app:toggleTodos/Transcript/Brief/Terminal/redraw (Ctrl+T, Ctrl+O, ...) | at hook×5 | Global keys and the exit states |
| chunk-3ghxrna3.js | `function eLe\(\{jumpRef:h,` | transcript search bar ("/" in transcript) | onKeyDown, onPaste, autoFocus | Transcript view |
| chunk-3ghxrna3.js | `function Jne\(h\)\{let Ce=w` | Transcript context (Ctrl+E show all, Esc/q/Ctrl-C exit) | at hook×2 | Transcript view |
| chunk-3ghxrna3.js | `function sG\(h\)\{let go=w\(` | transcript screen focus | focus hook | Transcript view |
| chunk-3ghxrna3.js | `Auto mode works better when it knows your environment\. Takes about a m` | environment onboarding offer ("Teach auto mode about your environment?") | onKeyDown | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Auto mode works better when it knows your environment\. Takes about a m` | env onboarding / fullscreen_upsell select | Select, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Try the new fullscreen renderer\?` | fullscreen_upsell ("Try the new fullscreen renderer?") | Confirm, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Switching back to the classic renderer\\u2026` | /tui switch-back feedback ("Switching back to the classic renderer…") | at hook, TextInput, Dialog frame | Slash-command panels |
| chunk-3ghxrna3.js | `function bG\(h\)\{let Se=w\(` | Chat-context command: keybindings (user keybindings that run slash commands) while the prompt is focused | on hook | The prompt input |
| chunk-3ghxrna3.js | `Rewinding does not affect files edited manually or via bash\.` | rewind/restore confirmation (Settings ctx confirm:no, app:interrupt) | at hook×2 | Message selector (Rewind) |
| chunk-3ghxrna3.js | `Message not found\.` | message selector (registerActiveContext) | active context, exit keys | Message selector (Rewind) |
| chunk-3ghxrna3.js | `Nothing to rewind to yet\.` | rewind restore options select | Select, Dialog frame, List | Message selector (Rewind) |
| chunk-3ghxrna3.js | `Pending wakeup cancelled` | Chat-context bindings helper (footer may own a key) | on hook | The prompt input |
| chunk-3ghxrna3.js | `No background agents running` | chat:cancel (Esc), app:interrupt (Ctrl-C), app:exit (Ctrl-D), chat:killAgents while the prompt is mounted | at hook×4 | Global keys and the exit states |
| chunk-3ghxrna3.js | `function Zm\(\{inputValue:` | digit-answer pre-dispatch for inline prompts (chat:submit guard) | pre-dispatch | The prompt input |
| chunk-3ghxrna3.js | `function AD\(h\)\{let ne=w\(` | above-prompt band buttons (onAction pane:scroll, abovePrompt:next/previous) | onAction×4, Pressable×2 | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `function Rae\(h,x,L\)\{let ` | above-prompt leave | pre-dispatch | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `function hz\(h\)\{let x=ud\(` | above-prompt pre-dispatch | pre-dispatch | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `function nr\(\$n\)\{lsn\.set\(` | above-prompt band (Chat ctrl+x ctrl+a toggle, ctrl+x tab focus; AbovePrompt* contexts; va above-prompt-*) | at hook×2, active context×3, overlay×2 | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `function Fs\(\)\{let \$n=\$e\(` | above-prompt contexts | on hook×3, pre-dispatch | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `function mUt\(Z\)\{return Z` | above-prompt onAction | onAction | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `function lle\(h\)\{let\{pane` | pane chat:cancel | at hook | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `function Qn\(Ts\)\{let Ps=a` | pane (Pane/PaneField contexts) | active context×4, overlay×2 | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `function oi\(\)\{let Ts=yn\(` | pane bindings | on hook×5, pre-dispatch | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `Fast mode was not changed: applying it failed` | /fast picker (va fast-mode-picker; also opened from footer) | overlay | Slash-command panels |
| chunk-3ghxrna3.js | `Running PreModelSwitch hooks\\u2026` | /fast confirmation (Confirmation ctx) | on hook, Dialog frame | Slash-command panels |
| chunk-3ghxrna3.js | `Fast mode is not available` | /fast | FastModePicker | Slash-command panels |
| chunk-3ghxrna3.js | `function sde\(h,x\)\{let\{st` | Ctrl+R history search (Global history:search; HistorySearch context) | at hook, on hook | History search |
| chunk-3ghxrna3.js | `function jde\(\{draft:h,co` | autocomplete suggestions (Autocomplete context; va autocomplete) | on hook, active context, overlay | Autocomplete |
| chunk-3ghxrna3.js | `Remote Control` | Remote Control panel rendered in place of the editor | on hook, overlay | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Remote Control` | Remote Control panel keys | onKeyDown | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Still checking the last artifact you chose\.` | app:openArtifact (Ctrl+]) opens artifacts picker | at hook | Global keys and the exit states |
| chunk-3ghxrna3.js | `Artifacts in this session` | artifacts picker rendered in place of the editor (va artifacts-dialog) | overlay, Dialog frame, List | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Type to search\\u2026` | shared fuzzy picker (artifacts, history picker, theme tokens) | onKeyDown, onPaste, autoFocus, SearchBox | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Search prompts ` | fuzzy history picker ("Search prompts", HistorySearch ctrl+s scope) | at hook, overlay, FuzzyPicker | History search |
| chunk-3ghxrna3.js | `Current model` | /model picker search | on hook | Slash-command panels |
| chunk-3ghxrna3.js | `Current model` | /model picker (ModelPicker context) | on hook | Slash-command panels |
| chunk-3ghxrna3.js | `Select model` | /model picker exit handling | exit keys | Slash-command panels |
| chunk-3ghxrna3.js | `Switch between Claude models\. Your pick becomes the default for new se` | /model picker select | Select, onKeyDown, onPaste, autoFocus | Slash-command panels |
| chunk-3ghxrna3.js | `Claude will think before responding` | thinking toggle dialog (Alt+T chat:thinkingToggle) | at hook×2 | Slash-command panels |
| chunk-3ghxrna3.js | `Changing thinking mode mid-conversation will increase latency and may ` | thinking toggle mid-conversation confirm | Select | Slash-command panels |
| chunk-3ghxrna3.js | `Ultrareview stopped\. Run /ultrareview to start a new review, which use` | proactivity menu (ProactivityMenu context) | overlay | Slash-command panels |
| chunk-3ghxrna3.js | `Ultrareview stopped\. Run /ultrareview to start a new review, which use` | proactivity menu | on hook×2, pre-dispatch | Slash-command panels |
| chunk-3ghxrna3.js | `Async agent` | background tasks dialog: async agent detail | onKeyDown, detail keys, Dialog frame, KeyBox | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `The scan ended without producing a proposal\.` | background tasks dialog: auto-mode scan detail | onKeyDown, detail keys, Dialog frame, KeyBox | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Memory consolidation` | background tasks dialog: memory consolidation detail | onKeyDown, detail keys | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Review the plan in your browser` | background tasks dialog: cloud session detail | onKeyDown, detail keys, Dialog frame, KeyBox | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Stop ultraplan` | background tasks dialog: stop ultraplan | Select, Dialog frame | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Terminate session and discard plan` | background tasks dialog: stop ultraplan confirm | Confirm, Dialog frame | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `This archives the cloud session and stops local tracking\. The review w` | background tasks dialog: stop ultrareview confirm | Confirm, Dialog frame | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Stop ultrareview` | background tasks dialog: ultrareview actions | Select, Dialog frame | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Cloud session details` | background tasks dialog: cloud session details | onKeyDown, Dialog frame, KeyBox | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Shell details` | background tasks dialog: shell details | onKeyDown, detail keys, Dialog frame, KeyBox | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `function IKt\(h\)\{let Mt=w` | background tasks dialog (va background-tasks-dialog) | overlay | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Background dialog dismissed` | background tasks dialog list | Select, Dialog frame, List | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Background dialog dismissed` | cloud tasks dialog (va cloud-tasks-dialog) | overlay | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Running in the cloud session` | tasks dialog list | Select, onKeyDown, Dialog frame, KeyBox | Overlays drawn in place of the editor |
| chunk-3ghxrna3.js | `Prompt from an external link` | pre-filled prompt from an external link confirmation | Confirm, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `function oYe\(h\)\{let Se=w` | Ctrl+R history search input ("search prompts:") | TextInput | History search |
| chunk-3ghxrna3.js | `VISUAL LINE` | prompt editor TextInput (oAt) in vim mode | TextInput, onPaste, TextInput | The prompt input |
| chunk-3ghxrna3.js | `Your next response will be slower and use more tokens` | /model switch cache warning ("No, go back") | Confirm, Dialog frame | Slash-command panels |
| chunk-3ghxrna3.js | `History search isn't available in cloud sessions yet` | prompt input key handling: chat:submit registerHandler, Help (?), history:search, chat:fastMode | at hook×2, registerHandler | The prompt input |
| chunk-3ghxrna3.js | `function p_\(\)\{Bc\(Gu,-1\)\}` | footer selection (Footer context) and footer-opened dialogs (tasks, fast, workflows) | on hook, onKeyDown, autoFocus, FastModePicker, TasksDialog, CloudTasksDialog, WorkflowDetail | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `Couldn't send feedback\.` | Claude-drafted feedback panel (chat:cancel) | at hook | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `Claude recalled a memory:` | memory recall survey (digit answers) | onAction, Pressable | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `Claude recalled a memory:` | transcript share survey | pre-dispatch, overlay, TextInput | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `Thanks for the feedback!` | feedback survey thanks | onAction, Pressable | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `Sending feedback` | feedback survey (FeedbackSurvey ctrl+x 1/2/3) | on hook, onAction, Pressable | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `Feedback: ` | feedback survey | pre-dispatch | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `Survey from ` | plugin survey offer | onAction×2 | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `You've used its skills recently` | plugin usage survey | onAction×2, Pressable | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `\(Select all that apply\)` | plugin survey (Tabs) | on hook, pre-dispatch×2 | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `\(Select all that apply\)` | plugin survey question | autoFocus | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `\(Select all that apply\)` | plugin survey question select | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Review your answers` | plugin survey free-text answer | TextInput | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `This survey is from ` | plugin survey submit | Confirm | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `function Vke\(h\)\{let Z=w\(` | inline digit question above the prompt (Chat chat:cancel when input empty) | at hook, onAction, Pressable | Above-prompt band, panes, footer selection and surveys |
| chunk-3ghxrna3.js | `Yes, set auto mode as my default permission mode` | auto_default_nudge | exit keys | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Yes, set auto mode as my default permission mode` | auto_default_nudge | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `function ave\(h\)\{let Z=w\(` | auto_mode_server_fallback notice (bE Confirmation) | scoped bindings, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `function uve\(h\)\{let Se=w` | plugin_hint / nudge exit handling | exit keys | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `\(unnamed plugin\)` | plugin_hint | Confirm | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `\(unnamed command\)` | plugin_hint install | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `We recommend compacting earlier` | autocompact_offer | pre-dispatch | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `function Lve\(h\)\{let Le=w` | cost_threshold exit handling | exit keys | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Learn more about how to monitor your spending:` | cost_threshold | pre-dispatch, Select, onPaste, autoFocus | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Learn more about how to monitor your spending:` | cost_threshold | Select, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Turning off auto-reload\\u2026` | /usage-credits | onAction | Slash-command panels |
| chunk-3ghxrna3.js | `Usage credits` | /usage-credits | on hook, Dialog frame | Slash-command panels |
| chunk-3ghxrna3.js | `Out of usage credits \\u2014 buy more below to keep going\.` | /usage-credits | Select, onAction, Dialog frame | Slash-command panels |
| chunk-3ghxrna3.js | `By confirming, you allow Anthropic to charge your card in the amount a` | /usage-credits buy | Select, Dialog frame | Slash-command panels |
| chunk-3ghxrna3.js | `Buy usage credits for` | /usage-credits buy | Select, Dialog frame | Slash-command panels |
| chunk-3ghxrna3.js | `Confirm amount` | /usage-credits limit | on hook, TextInput, Dialog frame | Slash-command panels |
| chunk-3ghxrna3.js | `Monthly limit: ` | /usage-credits limit | onKeyDown, TextInput | Slash-command panels |
| chunk-3ghxrna3.js | `Payment failed` | /usage-credits payment | onKeyDown, TextInput×2 | Slash-command panels |
| chunk-3ghxrna3.js | `Payment failed` | /usage-credits payment | TextInput, Dialog frame | Slash-command panels |
| chunk-3ghxrna3.js | `Request usage credits from your admin` | /usage-credits admin request | Confirm, Dialog frame | Slash-command panels |
| chunk-3ghxrna3.js | `You don't have usage credits yet\.` | /usage-credits | Select, Dialog frame×5 | Slash-command panels |
| chunk-3ghxrna3.js | `Welcome to Claude Code for ` | ide_onboarding | on hook, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `The session keeps running on this machine\. Use your other devices as a` | resume_return / cold_resume_quota | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Resume full session as-is` | resume_return | Select, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Resume this conversation\?` | cold_resume_quota ("Resume this conversation?") | pre-dispatch, Select, onPaste, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `You are a date/time parser that converts natural language into ISO 860` | MCP elicitation form date field | Select, Dialog frame | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `This field is required` | MCP elicitation form (va elicitation) | overlay | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `This field is required` | MCP elicitation form | at hook | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `MCP server \\u201C` | MCP elicitation | onKeyDown, Dialog frame, KeyBox | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Continue without waiting` | MCP elicitation waiting (va elicitation-url) | overlay | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Continue without waiting` | MCP URL elicitation | onKeyDown×2, Dialog frame×2, KeyBox×2 | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Type something\\u2026` | MCP elicitation form fields | TextInput | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Screen Recording:` | computer_use_approval (macOS permissions) | Select, Dialog frame | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `\(already granted\)` | computer_use_approval | Select, Dialog frame | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `No, and tell Claude what to do differently ` | sandbox_network_access | Select | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Implement here` | ultraplan_choice (va ultraplan-choice) | overlay | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Clear conversation and start with only the plan` | ultraplan_choice | Select, onKeyDown, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Run ultraplan in the cloud\?` | ultraplan_launch (va ultraplan-launch) | overlay | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Run ultraplan in the cloud\?` | ultraplan_launch | Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Run ultraplan` | ultraplan_launch | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `No Python package manager found \(uvx, pipx, or pip\)` | it2_setup | at hook | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Requires Python \(uvx, pipx, or pip\)` | it2_setup | onKeyDown, autoFocus | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `To use native iTerm2 split panes for teammates, you need the` | it2_setup | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Installation failed` | it2_setup | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Verification failed` | it2_setup | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `No preview available` | AskUserQuestion (Tabs, chat:externalEditor) | at hook, on hook, onKeyDown, autoFocus, TextInput | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Planning: ` | AskUserQuestion question view | Select, onKeyDown, autoFocus | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `You have not answered all questions` | AskUserQuestion submit review | Confirm | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Pasted image` | AskUserQuestion free-text input (va ask-user-question-text-input) | overlay | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `function ll\(fi\)\{if\(fi===` | AskUserQuestion tabs | on hook | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Review rules that skip checks` | auto_mode_setup_review | Select, Dialog frame×2 | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Looks good \\u2014 save it` | auto_mode_setup_review | Select, Dialog frame | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Yes, and don\\u2019t ask again for` | permission frame bindings (app:toggleTranscript Ctrl+O in a permission dialog) | scoped bindings | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Allow all actions on ` | permission_browser | Select | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `The sending session's permission mode class doesn't match this session` | peer_inbound_approval | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Another Claude session sent a message: from ` | peer_inbound_approval / chrome_install_upsell | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Couldn't connect to the extension\. Finish setup later with /chrome\.` | chrome_install_setup | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `This task could use your Chrome browser\. The Claude in Chrome extensio` | chrome_install_upsell / permission_enter_plan_mode | Select | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `In plan mode, Claude will:` | permission_enter_plan_mode | Confirm | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Exit plan mode\?` | permission_exit_plan_mode_v2 | Select×2, onKeyDown, KeyBox | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Publish a plugin\?` | permission_publish_plugin | Dialog frame, List | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Publish a plugin\?` | permission_publish_plugin | scoped bindings, Confirm | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Save file to continue\\u2026` | goal_proposal | on hook, Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Claude has finished its current work \\u2014 approving starts it workin` | goal_proposal confirm | Confirm | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Open WebSocket ` | generic/monitor permission via LZ | PermissionPrompt | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `PowerShell command` | permission_powershell (and refusal fallback options) | Select | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Session paused` | refusal_fallback_prompt ("Session paused") | Select | Other dialogs in the dialog store |
| chunk-3ghxrna3.js | `Use this skill\?` | permission_skill | PermissionPrompt | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `No, and tell Claude what to do differently ` | permission_webfetch | Select | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `function eF\(h\)\{let Ht=w\(` | REPL layout box (onKeyDownCapture/onKeyDown for fullscreen scroll and vim) | onKeyDown, onKeyDownCapture | None: takes no Return or text in place of the prompt input |
| chunk-3ghxrna3.js | `function aAe\(h\)\{let \$e=w` | dialog host pre-dispatch (auto-deny presence) | pre-dispatch | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-3ghxrna3.js | `Exit and stop tasks` | exit flow ("Exit and stop tasks / Move to background and exit") | Select | Global keys and the exit states |
| chunk-3ghxrna3.js | `Keep worktree` | exit worktree prompt (Keep/Remove worktree) | Select, Dialog frame | Global keys and the exit states |
| chunk-3ghxrna3.js | `function qMe\(\)\{let\{inter` | suspend/resume (Ctrl+Z) notice | app event listener | None: takes no Return or text in place of the prompt input |
| chunk-3ghxrna3.js | `Diff unavailable` | app:toggleReplTab | at hook | Global keys and the exit states |
| chunk-3ghxrna3.js | `Diff unavailable` | diff panel (DiffPanel context, app:diffFileList*) | at hook×2 | none: no Return or text binding |
| chunk-3ghxrna3.js | `Too many changed files to show diff` | diff panel | at hook, on hook | Above-prompt band, panes, footer selection and surveys |
| chunk-3jxq5p27.js | `Do you want to use this API key\?` | custom API key approval | Confirm, Dialog frame | Startup screens |
| chunk-3kb5ba6v.js | `function Z\(c\)\{let L=Dc\(t` | shared List/scroll-list component (rn/od) with select:* bindings; each use listed | on hook×2 | How a key travels (shared component: each use is listed under its own row) |
| chunk-3kb5ba6v.js | `function Ae\(c\)\{ne\.pin\(\),` | shared List/scroll-list component (rn/od) with select:* bindings; each use listed | onKeyDown | How a key travels (shared component: each use is listed under its own row) |
| chunk-3kb5ba6v.js | `function Sn\(o\)\{let m=w\(2` | shared List/scroll-list component (rn/od) with select:* bindings; each use listed | Select (screen reader), onKeyDown | How a key travels (shared component: each use is listed under its own row) |
| chunk-3kb5ba6v.js | `function Je\(\{props:\{chil` | shared List/scroll-list component (rn/od) with select:* bindings; each use listed | focus hook | How a key travels (shared component: each use is listed under its own row) |
| chunk-3ncdx684.js | `Cloud gateway` | login / cloud gateway / provider setup wizard (also /login, /setup-*) | at hook×2, pre-dispatch, Confirm×2 | Startup screens |
| chunk-3ncdx684.js | `Cloud gateway \\xB7 sign in` | login / cloud gateway / provider setup wizard (also /login, /setup-*) | at hook | Startup screens |
| chunk-3ncdx684.js | `Press Esc to cancel` | login / cloud gateway / provider setup wizard (also /login, /setup-*) | at hook | Startup screens |
| chunk-3ncdx684.js | `Press Esc to go back` | login / cloud gateway / provider setup wizard (also /login, /setup-*) | at hook | Startup screens |
| chunk-3ncdx684.js | `Login method pre-selected: Subscription Plan \(Claude Pro/Max\)` | login / cloud gateway / provider setup wizard (also /login, /setup-*) | at hook×6 | Startup screens |
| chunk-3ncdx684.js | `Anthropic Console account \\xB7` | login / cloud gateway / provider setup wizard (also /login, /setup-*) | Select×3, TextInput | Startup screens |
| chunk-4336q1g5.js | `Updating spend limit\\u2026` | slash command /rate-limit-options | onKeyDown, Dialog frame, KeyBox | Slash-command panels |
| chunk-4336q1g5.js | `What do you want to do\?` | slash command /rate-limit-options | Select, onKeyDownCapture, onPasteCapture, Dialog frame×4 | Slash-command panels |
| chunk-48c1dh8j.js | `Resume cancelled` | slash command /resume | at hook | Slash-command panels |
| chunk-4byvvgs4.js | `Please use --channels to run a list of approved channels\.` | development channels warning | Confirm, Dialog frame | Startup screens |
| chunk-4hf49yd0.js | `function ye\(\)\{if\(F\.curre` | slash command /effort | exit keys | Slash-command panels |
| chunk-4hf49yd0.js | `Ultracode: ` | slash command /effort | on hook, onKeyDown, autoFocus | Slash-command panels |
| chunk-4hf49yd0.js | `function Ut\(d\)\{let t=w\(4` | slash command /effort | Select (screen reader) | Slash-command panels |
| chunk-4st36e9e.js | `function ua\(e\)\{return\{ge` | DOM shim attribute table (autoFocus is an HTML attribute name, no UI) | autoFocus | None: takes no Return or text in place of the prompt input |
| chunk-4z4prp8c.js | `function Y\(Z\)\{let a=w\(63` | background task detail (background tasks dialog) | onKeyDown, detail keys, Dialog frame, KeyBox | Overlays drawn in place of the editor |
| chunk-55qwcww5.js | `function Ge\(o\)\{let r=w\(3` | base Pressable (mI) and text-entry (Jse) components; each use listed | onKeyDown, onAction, autoFocus×2 | How a key travels (shared component: each use is listed under its own row) |
| chunk-55qwcww5.js | `function qe\(o\)\{let r=w\(3` | base Pressable (mI) and text-entry (Jse) components; each use listed | onKeyDown×2, onPaste×2, autoFocus×2, textEntry | How a key travels (shared component: each use is listed under its own row) |
| chunk-59apvq15.js | `function v3t\(\{handler:i,` | Task context task:background (Ctrl+B) while a foreground tool runs; acts on the running task, not a dialog answer | registerHandler | The prompt input (Keys at the prompt) |
| chunk-5ac3cnn0.js | `Pin model versions` | provider model pinning | exit keys | Startup screens |
| chunk-5ac3cnn0.js | `Pin model versions` | provider model pinning | Dialog frame | Startup screens |
| chunk-5ac3cnn0.js | `Pin model versions` | provider model pinning | Select | Startup screens |
| chunk-5ac3cnn0.js | `\(currently pinned\)` | provider model pinning | Select | Startup screens |
| chunk-5b3m5wsr.js | `No advisor` | slash command /advisor | Select, Dialog frame | Slash-command panels |
| chunk-5pqr5km1.js | `Yes covers those too, plus any this project adds later\.` | external CLAUDE.md imports approval (also /config) | Confirm, Dialog frame | Startup screens |
| chunk-5yfg4nye.js | `Paste code here if prompted > ` | 'claude auth login' readline ("Paste code here if prompted > ") | stdin listener, stdin data | None: not part of the interactive session the brain runs |
| chunk-5z8k4s4b.js | `Directory ` | /daemon scheduled-task editor (chunk-yybr850r) and conversation import | Select, Confirm, Dialog frame×2 | Slash-command panels |
| chunk-5z8k4s4b.js | `Save changes` | /daemon scheduled-task editor (chunk-yybr850r) and conversation import | FormDialog | Slash-command panels |
| chunk-6ewgphaj.js | `No \\u2014 ask me before each command` | remote spawn permission choice (jvmshzp2 launcher; also /daemon) | Select, Dialog frame | Startup screens |
| chunk-6kmhbk7e.js | `Loading Claude Code sessions\\u2026` | cloud session picker (/teleport, /remote) | at hook | Slash-command panels |
| chunk-6kmhbk7e.js | `Select a session to resume` | cloud session picker (/teleport, /remote) | Select, onKeyDown×4, autoFocus×4 | Slash-command panels |
| chunk-6kmhbk7e.js | `Resuming session\\u2026` | cloud session picker (/teleport, /remote) | at hook×2 | Slash-command panels |
| chunk-6medpmh3.js | `Restarting Claude Code\\u2026` | restart confirm (/setup-vertex, /setup-bedrock) | at hook | Slash-command panels |
| chunk-6zpfczgq.js | `Memory dialog dismissed` | slash command /memory | Dialog frame×2 | Slash-command panels |
| chunk-6zpfczgq.js | `Add a CLAUDE\.md to the repo, or ask Claude to write one` | slash command /memory | List | Slash-command panels |
| chunk-6zpfczgq.js | `Synced project memory: ` | slash command /memory | at hook, scoped bindings, Select, exit keys, List, Form | Slash-command panels |
| chunk-6zpfczgq.js | `Cancelled memory editing` | slash command /memory | Dialog frame | Slash-command panels |
| chunk-77me71j9.js | `Claude Desktop` | slash command /desktop | onKeyDown×2, Dialog frame×2, KeyBox×2 | Slash-command panels |
| chunk-7c7psh54.js | `Stop hook not found` | slash command /loops | onKeyDown, autoFocus, TextInput×2, Dialog frame, List | Slash-command panels |
| chunk-7mvcc9f1.js | `Teleport cancelled` | /teleport (chunk-93d692my) | exit keys | Slash-command panels |
| chunk-7mvcc9f1.js | `Moving your session\\u2026` | /teleport (chunk-93d692my) | Select | Slash-command panels |
| chunk-7repc4n1.js | `Last check` | slash command /goal | Dialog frame×3 | Slash-command panels |
| chunk-7ytmfe1t.js | `New sessions from this window don't start in ultracode` | agent view (FleetView, mounted by mountFleetView), a separate top-level screen, not the REPL the brain runs | onAction | None: not part of the interactive session the brain runs |
| chunk-7ytmfe1t.js | `Can't attach \\u2014 this session is running in another terminal` | agent view (FleetView, mounted by mountFleetView), a separate top-level screen, not the REPL the brain runs | onAction | None: not part of the interactive session the brain runs |
| chunk-7ytmfe1t.js | `Image attachments aren't available when replying to cloud sessions yet` | agent view (FleetView, mounted by mountFleetView), a separate top-level screen, not the REPL the brain runs | focus hook, onKeyDown, onKeyDownCapture, onPaste×2, autoFocus, SearchBox | None: not part of the interactive session the brain runs |
| chunk-7ytmfe1t.js | `No groups yet \\u2014 press ` | agent view (FleetView, mounted by mountFleetView), a separate top-level screen, not the REPL the brain runs | SearchBox | None: not part of the interactive session the brain runs |
| chunk-7ytmfe1t.js | `This session is being deleted \\u2014 reopen /resume once it finishes` | agent view (FleetView, mounted by mountFleetView), a separate top-level screen, not the REPL the brain runs | onAction | None: not part of the interactive session the brain runs |
| chunk-7ytmfe1t.js | `function ng\(\{onAction:h,` | agent view (FleetView, mounted by mountFleetView), a separate top-level screen, not the REPL the brain runs | focus hook, onKeyDown×2, onKeyDownCapture×2, onPaste×3, onAction×3, autoFocus×2 | None: not part of the interactive session the brain runs |
| chunk-7ytmfe1t.js | `Login successful` | agent view (FleetView, mounted by mountFleetView), a separate top-level screen, not the REPL the brain runs | onAction, stdin listener, stdin readable | None: not part of the interactive session the brain runs |
| chunk-862vpswd.js | `Skills come from the repo's \.claude/skills and the plugins the session` | slash command /skills | Dialog frame, List | Slash-command panels |
| chunk-862vpswd.js | `No changes` | slash command /skills | on hook×2, onKeyDown×2, onPaste×2, autoFocus, exit keys, Dialog frame, List, SearchBox | Slash-command panels |
| chunk-86zjhdmr.js | `Remote Control is disabled by your organization's policy\.` | remote-control CLI entry | stdin listener, stdin data | None: not part of the interactive session the brain runs |
| chunk-96hkqee3.js | `function de\(\{isDisabled:` | shared multi-select / option rows (uses uxe, va("multi-select")) | overlay | How a key travels (shared component: each use is listed under its own row) |
| chunk-96hkqee3.js | `function OF\(o\)\{let u=w\(1` | shared multi-select / option rows (uses uxe, va("multi-select")) | MultiSelect | How a key travels (shared component: each use is listed under its own row) |
| chunk-96hkqee3.js | `function me\(o\)\{let I=w\(6` | shared multi-select / option rows (uses uxe, va("multi-select")) | focus hook, onKeyDown, OptionRow | How a key travels (shared component: each use is listed under its own row) |
| chunk-983p1kbs.js | `To change this later, run /theme` | onboarding (theme, terminal setup) | exit keys | Startup screens |
| chunk-983p1kbs.js | `Option\+Enter for newlines and no audible bell` | onboarding (theme, terminal setup) | Confirm | Startup screens |
| chunk-983p1kbs.js | `Couldn't save Theme: ` | onboarding (theme, terminal setup) | on hook×2 | Startup screens |
| chunk-9a4nte2x.js | `Loading dynamic workflow history\\u2026` | slash command /workflows | overlay, Dialog frame | Slash-command panels |
| chunk-9a4nte2x.js | `Dynamic workflows` | slash command /workflows | Dialog frame, List, WorkflowDetail, WorkflowPanel | Slash-command panels |
| chunk-9a4nte2x.js | `The cloud session may have stopped, and its workflows too` | slash command /workflows | overlay | Slash-command panels |
| chunk-9m9x5rv6.js | `Exit and fix manually` | configuration error reset | Select, Dialog frame | Startup screens |
| chunk-a2xynw3j.js | `How would you like to handle this\?` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | Select, Dialog frame | Slash-command panels |
| chunk-a2xynw3j.js | `This changes how Claude Code communicates with you` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | Dialog frame | Slash-command panels |
| chunk-a2xynw3j.js | `Enter your preferred response and voice language:` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | List | Slash-command panels |
| chunk-a2xynw3j.js | `Enter your preferred response and voice language:` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | at hook | Slash-command panels |
| chunk-a2xynw3j.js | `Enter your preferred response and voice language:` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | TextInput | Slash-command panels |
| chunk-a2xynw3j.js | `Notify when Claude needs you` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | Dialog frame, Form | Slash-command panels |
| chunk-a2xynw3j.js | `Reset Remote Control to default` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | at hook, on hook, focus hook, Select×2, onKeyDown, onPaste, autoFocus, Dialog frame×3, FormDialog, Form×2, SearchBox | Slash-command panels |
| chunk-a2xynw3j.js | `Scanning local sessions\\u2026` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | on hook | Slash-command panels |
| chunk-a2xynw3j.js | `Nothing over ` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | on hook | Slash-command panels |
| chunk-a2xynw3j.js | `Could not refresh usage data` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | at hook | Slash-command panels |
| chunk-a2xynw3j.js | `Stats dialog dismissed` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | at hook | Slash-command panels |
| chunk-a2xynw3j.js | `Failed to load stats: ` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | onKeyDown×4, autoFocus×4, Tabs | Slash-command panels |
| chunk-a2xynw3j.js | `No model usage data available` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | onKeyDown, autoFocus | Slash-command panels |
| chunk-a2xynw3j.js | `Settings dialog dismissed` | /config, /status, /usage settings panels (reached from chunk-6xw89mvj /config, chunk-6fdywzpj /status, chunk-z5qgfsp8 /usage, chunk-d8cht8hw /output-style /vim) | at hook, exit keys, Tabs | Slash-command panels |
| chunk-a4px3pma.js | `Yes, run hooks for cloud sessions` | slash command /hooks | Select, Dialog frame | Slash-command panels |
| chunk-a4px3pma.js | `Hook events` | slash command /hooks | Dialog frame, List | Slash-command panels |
| chunk-a4px3pma.js | `To add hooks, edit settings\.json directly or ask Claude` | slash command /hooks | Dialog frame×2, List | Slash-command panels |
| chunk-a4px3pma.js | `To add hooks, edit settings\.json directly or ask Claude` | slash command /hooks | Dialog frame×2, List | Slash-command panels |
| chunk-a4px3pma.js | `Status message:` | slash command /hooks | Dialog frame | Slash-command panels |
| chunk-a4px3pma.js | `Hooks dialog dismissed` | slash command /hooks | at hook×5, Select, Dialog frame | Slash-command panels |
| chunk-ac482sdk.js | `No TTY available to prompt for client secret\. Set MCP_CLIENT_SECRET en` | readline prompt for MCP OAuth client secret (CLI 'mcp add') | stdin listener, stdin data | None: not part of the interactive session the brain runs |
| chunk-ac9nfdnf.js | `Run a dynamic workflow\?` | permission_workflow ("Run a dynamic workflow?") via LZ | PermissionPrompt, onKeyDown, autoFocus | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-amjw57qh.js | `Enter some text, or Escape for the list\.` | shared Select (Ye/ns/zZ), multi-select (Eur), y/n confirm (vur), TextInput (oAt); each use listed | focus hook, overlay, onKeyDown | How a key travels (shared component: each use is listed under its own row) |
| chunk-amjw57qh.js | `Enter some text, or Escape for the list\.` | shared Select (Ye/ns/zZ), multi-select (Eur), y/n confirm (vur), TextInput (oAt); each use listed | focus hook, overlay, onKeyDown | How a key travels (shared component: each use is listed under its own row) |
| chunk-amjw57qh.js | `Please answer y or n\.` | shared Select (Ye/ns/zZ), multi-select (Eur), y/n confirm (vur), TextInput (oAt); each use listed | focus hook, overlay | How a key travels (shared component: each use is listed under its own row) |
| chunk-amjw57qh.js | `Please answer y or n\.` | shared Select (Ye/ns/zZ), multi-select (Eur), y/n confirm (vur), TextInput (oAt); each use listed | onKeyDown | How a key travels (shared component: each use is listed under its own row) |
| chunk-amjw57qh.js | `VISUAL LINE` | shared Select (Ye/ns/zZ), multi-select (Eur), y/n confirm (vur), TextInput (oAt); each use listed | TextInput | How a key travels (shared component: each use is listed under its own row) |
| chunk-amjw57qh.js | `function uxe\(t\)\{let T=w\(` | shared Select (Ye/ns/zZ), multi-select (Eur), y/n confirm (vur), TextInput (oAt); each use listed | at hook×3, on hook, onKeyDown, onPaste×2 | How a key travels (shared component: each use is listed under its own row) |
| chunk-amjw57qh.js | `function al\(t\)\{if\(To\(t\)\)` | shared Select (Ye/ns/zZ), multi-select (Eur), y/n confirm (vur), TextInput (oAt); each use listed | on hook, overlay | How a key travels (shared component: each use is listed under its own row) |
| chunk-amjw57qh.js | `function Ye\(t\)\{let a=w\(1` | shared Select (Ye/ns/zZ), multi-select (Eur), y/n confirm (vur), TextInput (oAt); each use listed | Select (screen reader), Select | How a key travels (shared component: each use is listed under its own row) |
| chunk-amjw57qh.js | `function ns\(t\)\{let E=w\(8` | shared Select (Ye/ns/zZ), multi-select (Eur), y/n confirm (vur), TextInput (oAt); each use listed | focus hook, onKeyDown, textEntry, OptionRow×3 | How a key travels (shared component: each use is listed under its own row) |
| chunk-atrng9mj.js | `Generate /import-to-claude-code skill to finish these interactively` | slash command /import | Dialog frame×2 | Slash-command panels |
| chunk-b67b1g5t.js | `Checking again\\u2026` | org-config blocking screen | Select, Dialog frame | Startup screens |
| chunk-ba9npfvn.js | `function jyi\(e,t\)\{if\(!e\)` | eval mock MCP server process | stdin listener×2 | None: not part of the interactive session the brain runs |
| chunk-bb4pkja0.js | `Checking credentials\\u2026` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | Dialog frame | Slash-command panels |
| chunk-bb4pkja0.js | `Not connected on claude\.ai yet for the account Claude Code is signed i` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | at hook×3, exit keys | Slash-command panels |
| chunk-bb4pkja0.js | `Clear authentication` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | Select, onKeyDown×3, autoFocus×3, TextInput, Dialog frame | Slash-command panels |
| chunk-bb4pkja0.js | `This may take a few moments\.` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | Select, Dialog frame×2 | Slash-command panels |
| chunk-bb4pkja0.js | `Full name: ` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | Dialog frame | Slash-command panels |
| chunk-bb4pkja0.js | `No tools available` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | Dialog frame, List | Slash-command panels |
| chunk-bb4pkja0.js | `Add marketplace\?` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | onKeyDown, TextInput, Dialog frame, KeyBox | Slash-command panels |
| chunk-bb4pkja0.js | `MCP Servers` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | at hook | Slash-command panels |
| chunk-bb4pkja0.js | `Community managed: maintained by its publisher, not Anthropic` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | Select | Slash-command panels |
| chunk-bb4pkja0.js | `function gn\(dt\)\{if\(ge\)\{v` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | on hook×2 | Slash-command panels |
| chunk-bb4pkja0.js | `Failed to load marketplaces` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | at hook | Slash-command panels |
| chunk-bb4pkja0.js | `All plugins from this marketplace are already installed\.` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | onKeyDown, onPaste, autoFocus, List, SearchBox | Slash-command panels |
| chunk-bb4pkja0.js | `function Nl\(\{screen:a,on` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | on hook | Slash-command panels |
| chunk-bb4pkja0.js | `Failed to load plugins` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | at hook×2, onKeyDown, onPaste, autoFocus, SearchBox | Slash-command panels |
| chunk-bb4pkja0.js | `This command came from a link:` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | onKeyDown, Dialog frame, KeyBox | Slash-command panels |
| chunk-bb4pkja0.js | `\(and the others\) ` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | at hook×3, on hook | Slash-command panels |
| chunk-bb4pkja0.js | `Manage marketplaces` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | onKeyDown×3, KeyBox×3 | Slash-command panels |
| chunk-bb4pkja0.js | `Run /reload-plugins to apply\.` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | at hook | Slash-command panels |
| chunk-bb4pkja0.js | `Disable plugin` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | on hook×4 | Slash-command panels |
| chunk-bb4pkja0.js | `Removed from marketplace \\xB7 reason: ` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | onKeyDown×6, onPaste, autoFocus, List, KeyBox×5, SearchBox | Slash-command panels |
| chunk-bb4pkja0.js | `Skills loaded this session` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | at hook | Slash-command panels |
| chunk-bb4pkja0.js | `No plugin errors` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | at hook, on hook | Slash-command panels |
| chunk-bb4pkja0.js | `No plugin errors` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | onKeyDown, KeyBox | Slash-command panels |
| chunk-bb4pkja0.js | `Stats tab rendered without getSkillStatsInputs` | /plugin and /mcp panels (reached from chunk-7dzqa5tp /plugin, chunk-cpnz1vqv /mcp, chunk-862vpswd /skills, chunk-g2qyjnfd /skill-doctor) | at hook, exit keys, Tabs | Slash-command panels |
| chunk-bj5jt8h4.js | `function Ct\(o\)\{let\{plugi` | plugin-rendered UI (buttons, inputs, keybinding fallbacks for plugin actions) in plugin panes | registerHandler | Above-prompt band, panes, footer selection and surveys |
| chunk-bj5jt8h4.js | `function Dt\(o\)\{let\{plugi` | plugin-rendered UI (buttons, inputs, keybinding fallbacks for plugin actions) in plugin panes | pre-dispatch, TextInput | Above-prompt band, panes, footer selection and surveys |
| chunk-bj5jt8h4.js | `function or\(\{plugin:o,ha` | plugin-rendered UI (buttons, inputs, keybinding fallbacks for plugin actions) in plugin panes | onAction, Pressable | Above-prompt band, panes, footer selection and surveys |
| chunk-bj5jt8h4.js | `function Co\(o\)\{let i=le\(` | plugin-rendered UI (buttons, inputs, keybinding fallbacks for plugin actions) in plugin panes | pre-dispatch | Above-prompt band, panes, footer selection and surveys |
| chunk-c3jpg58b.js | `Claude Code uses the standard GCP credential chain\. Pick the method yo` | Vertex provider setup (also /setup-vertex) | Select | Startup screens |
| chunk-c3jpg58b.js | `These will be written to ` | Vertex provider setup (also /setup-vertex) | Confirm | Startup screens |
| chunk-c3jpg58b.js | `Type a different project\\u2026` | Vertex provider setup (also /setup-vertex) | at hook, TextInput, List | Startup screens |
| chunk-c3jpg58b.js | `Region is required` | Vertex provider setup (also /setup-vertex) | at hook, TextInput | Startup screens |
| chunk-c3jpg58b.js | `Path is required` | Vertex provider setup (also /setup-vertex) | at hook, TextInput | Startup screens |
| chunk-c3jpg58b.js | `This may take a few seconds\.` | Vertex provider setup (also /setup-vertex) | Select, Confirm | Startup screens |
| chunk-cfsh2fth.js | `Claude Code will restart to apply\.` | model pin update offer | Confirm, Dialog frame | Startup screens |
| chunk-cgpfw3jq.js | `You can also configure this in /config or with the --ide flag` | slash command /ide | Select, Dialog frame | Slash-command panels |
| chunk-cgpfw3jq.js | `Do you wish to disable auto-connect to IDE\?` | slash command /ide | Confirm, Dialog frame | Slash-command panels |
| chunk-cgpfw3jq.js | `No available IDEs detected\. Make sure your IDE has the Claude Code ext` | slash command /ide | List | Slash-command panels |
| chunk-cgpfw3jq.js | `IDE selection cancelled` | slash command /ide | Dialog frame, List | Slash-command panels |
| chunk-cgpfw3jq.js | `IDE selection cancelled` | slash command /ide | Dialog frame, List | Slash-command panels |
| chunk-cpb453mc.js | `function at\(e,o,r=\{\}\)\{le` | definitions of at/on/Kf hooks | registerHandler | How a key travels (shared component: each use is listed under its own row) |
| chunk-cpb453mc.js | `function on\(e,o=\{\}\)\{let\{` | definitions of at/on/Kf hooks | registerHandler | How a key travels (shared component: each use is listed under its own row) |
| chunk-cpb453mc.js | `function Kf\(e,\{isActive:` | definitions of at/on/Kf hooks | registerPreDispatch | How a key travels (shared component: each use is listed under its own row) |
| chunk-cpnz1vqv.js | `Authenticating with ` | slash command /mcp | at hook, Select, Dialog frame | Slash-command panels |
| chunk-cpnz1vqv.js | `Show unused connectors` | slash command /mcp | Dialog frame, List | Slash-command panels |
| chunk-cshb3rdc.js | `Binary file - cannot display diff` | diff panel view (fullscreen side panel) | autoFocus | Above-prompt band, panes, footer selection and surveys |
| chunk-ct1wasm0.js | `Guest passes dialog dismissed` | slash command /passes | at hook, autoFocus, exit keys | Slash-command panels |
| chunk-ct1wasm0.js | `Referral link copied to clipboard!` | slash command /passes | onKeyDown×2, KeyBox | Slash-command panels |
| chunk-cynn13e7.js | `Do you want to proceed\?` | shared permission prompt LZ ("Do you want to proceed?") and its frame | on hook, Select | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-cynn13e7.js | `About the ` | shared permission prompt LZ ("Do you want to proceed?") and its frame | scoped bindings, PermissionPrompt×2 | Permission dialogs; AskUserQuestion; Plan mode; MCP elicitation |
| chunk-d8fzwfwq.js | `function bE\(\{scope:n,bin` | bE scoped-binding box definition | onAction, keybindingScope | How a key travels (shared component: each use is listed under its own row) |
| chunk-db2vg3tt.js | `Select a project to purge:` | CLI subcommand project purge picker | Select | None: not part of the interactive session the brain runs |
| chunk-ddk45ec3.js | `Resetting your session limit\\u2026 this usually takes a few seconds` | slash command /limit-reset | Dialog frame | Slash-command panels |
| chunk-dp324v95.js | `os\(\(\)=>\{if` | settings-panel exit helper ZRe (used by /status, /config, /usage and device/remote dialogs) | exit keys | Slash-command panels |
| chunk-e4p6x208.js | `CSI \? u not asked` | InternalApp raw stdin reader, key parser, focus dispatch | app event listener×6 | How a key travels (shared component: each use is listed under its own row) |
| chunk-e4p6x208.js | `Tree: only a text or a span has text` | InternalApp raw stdin reader, key parser, focus dispatch | app event listener | How a key travels (shared component: each use is listed under its own row) |
| chunk-e6d6s4y7.js | `Your Pro plan includes a Claude Code trial\.` | Pro plan trial offer | on hook | Startup screens |
| chunk-e9xpztv6.js | `Did not add a working directory\.` | slash command /add-dir | KeyBox×2 | Slash-command panels |
| chunk-es652crg.js | `New custom theme` | slash command /theme | at hook, TextInput×2, FuzzyPicker | Slash-command panels |
| chunk-evaj6163.js | `function Yb\(d\)\{let h=w\(5` | shared Tabs container Yb (Tabs context); each use listed | on hook×2, onKeyDown×2, autoFocus×2 | How a key travels (shared component: each use is listed under its own row) |
| chunk-ex353g6q.js | `\(unknown path\)` | dynamic workflow / agents detail dialogs (/workflows; Wot also from the footer) | at hook, onKeyDown, TextInput, Dialog frame, KeyBox | Slash-command panels |
| chunk-ex353g6q.js | `function Wot\(\{workflow:o` | dynamic workflow / agents detail dialogs (/workflows; Wot also from the footer) | overlay | Slash-command panels |
| chunk-ex353g6q.js | `No agents yet\.` | dynamic workflow / agents detail dialogs (/workflows; Wot also from the footer) | on hook | Slash-command panels |
| chunk-ex353g6q.js | `No agents yet\.` | dynamic workflow / agents detail dialogs (/workflows; Wot also from the footer) | onKeyDown, Dialog frame, WorkflowPanel, KeyBox | Slash-command panels |
| chunk-ezyd5fyv.js | `If you piped input, it was not received \\u2014 pass it as a prompt arg` | piped stdin read for -p/--print | stdin listener, stdin data | None: not part of the interactive session the brain runs |
| chunk-fam06ba5.js | `function hR\(m\)\{let R=w\(2` | hR exit/interrupt helper used by blocking screens (org-config, managed settings, permissions, usage) | exit keys | Global keys and the exit states |
| chunk-fgyxav5k.js | `function Zwi\(\)\{let e=i\(\)` | early input capture before Ink takes stdin: typed text buffered, CR kept as newline, Ctrl-C exits 130; seeded into the prompt when the REPL mounts | stdin listener, stdin readable | Startup screens |
| chunk-gq2vt045.js | `Catch you later!` | slash command /exit | onKeyDown, KeyBox | Slash-command panels |
| chunk-he8cf6we.js | `Select a version to view its notes\.` | slash command /release-notes | Dialog frame, List | Slash-command panels |
| chunk-hen6700m.js | `No changes detected` | teleport login/stash (/teleport; also --teleport at startup) | Confirm, Dialog frame | Slash-command panels |
| chunk-hen6700m.js | `Teleport requires a Claude\.ai account\.` | teleport login/stash (/teleport; also --teleport at startup) | at hook, scoped bindings, Confirm, exit keys, Dialog frame | Slash-command panels |
| chunk-hqrt52d9.js | `Settings Error` | invalid settings dialog (launchInvalidSettingsDialog) | Select, Dialog frame | Startup screens |
| chunk-hsw930hd.js | `More workflow examples \(issue triage, CI fixes, etc\.\) at:` | slash command /install-github-app | Dialog frame | Slash-command panels |
| chunk-hsw930hd.js | `Use your existing Claude Code API key` | slash command /install-github-app | on hook×2, onPaste, TextInput | Slash-command panels |
| chunk-hsw930hd.js | `ANTHROPIC_API_KEY already exists in repository secrets!` | slash command /install-github-app | on hook×2, TextInput | Slash-command panels |
| chunk-hsw930hd.js | `Use current repository: ` | slash command /install-github-app | on hook×2, TextInput | Slash-command panels |
| chunk-hsw930hd.js | `A Claude workflow file already exists at` | slash command /install-github-app | Select | Slash-command panels |
| chunk-hsw930hd.js | `Install the Claude GitHub App` | slash command /install-github-app | at hook | Slash-command panels |
| chunk-hsw930hd.js | `Creating a long-lived token for GitHub Actions` | slash command /install-github-app | onKeyDown, KeyBox | Slash-command panels |
| chunk-hsw930hd.js | `Starting authentication\\u2026` | slash command /install-github-app | TextInput | Slash-command panels |
| chunk-hsw930hd.js | `GitHub App installed!` | slash command /install-github-app | Select | Slash-command panels |
| chunk-hsw930hd.js | `We found some potential issues, but you can continue anyway` | slash command /install-github-app | at hook | Slash-command panels |
| chunk-hsw930hd.js | `Installation cancelled by user` | slash command /install-github-app | exit keys | Slash-command panels |
| chunk-hsw930hd.js | `Checking for an existing ANTHROPIC_API_KEY secret\\u2026` | slash command /install-github-app | onKeyDown×2, autoFocus×2, Dialog frame | Slash-command panels |
| chunk-hwx61baj.js | `function it registered f` | zod schema text for plugin UI elements (autoFocus is a schema field, no registration) | autoFocus×3 | None: takes no Return or text in place of the prompt input |
| chunk-hz53hyjd.js | `function j\(o\)\{let e=awai` | Computer Use MCP server process | stdin listener×2 | None: not part of the interactive session the brain runs |
| chunk-j79bfp5x.js | `Take the tour` | onboarding tour offer | Confirm | Startup screens |
| chunk-j84edmt2.js | `Auto \(match terminal\)` | theme picker (onboarding; also /theme) | at hook×3, active context, exit keys, List | Startup screens |
| chunk-jxftkk1f.js | `function f4t\(t\)\{let q=w\(` | shared text-input renderer f4t; each use listed | focus hook, onKeyDown, onPaste×2, autoFocus, TextEntry×2 | How a key travels (shared component: each use is listed under its own row) |
| chunk-jxftkk1f.js | `function Or\(t\)\{let p=w\(5` | shared text-input renderer f4t; each use listed | TextInput | How a key travels (shared component: each use is listed under its own row) |
| chunk-jy8kzjv7.js | `function pe\(\{children:r,` | Box host element (passes onKeyDown/onPaste/onAction/keybindingScope/textEntry/autoFocus to ink-box) | onKeyDown×2, onKeyDownCapture×2, onPaste×2, onPasteCapture×2, onAction×2, onActionCapture×2, keybindingScope×2, autoFocus×2, textEntry×2, app event listener | How a key travels (shared component: each use is listed under its own row) |
| chunk-jzsn2jdm.js | `Configure overrides` | slash command /sandbox | Select | Slash-command panels |
| chunk-jzsn2jdm.js | `Cannot block unix domain sockets \(see Dependencies tab\)` | slash command /sandbox | on hook, exit keys, Tabs | Slash-command panels |
| chunk-jzsn2jdm.js | `Cannot block unix domain sockets \(see Dependencies tab\)` | slash command /sandbox | Select | Slash-command panels |
| chunk-k0k7hd82.js | `Accessing workspace:` | workspace trust dialog | at hook, Confirm, exit keys | Startup screens |
| chunk-k219taea.js | `No TTY available to prompt for client secret\. Set MCP_CLIENT_SECRET en` | readline prompt for MCP OAuth client secret (CLI) | stdin listener, stdin data | None: not part of the interactive session the brain runs |
| chunk-k75ham60.js | `function jhi\(o,s\)\{if\(Re\(` | login method picker (also /login) | LoginPicker | Startup screens |
| chunk-k75ham60.js | `function YSe\(o\)\{let p=w\(` | login method picker (also /login) | at hook, exit keys, Dialog frame | Startup screens |
| chunk-k9gmavk8.js | `Cloud session` | slash command /session | at hook, exit keys | Slash-command panels |
| chunk-khk1ge41.js | `function MF\(n,t\)\{let h=w` | keybinding registry provider (registerHandler/registerActiveContext/registerPreDispatch) | active context×2 | How a key travels (shared component: each use is listed under its own row) |
| chunk-kk4ryvf0.js | `Requires the Chrome extension\. Get started at` | Claude in Chrome onboarding | onKeyDown, autoFocus, Dialog frame | Startup screens |
| chunk-kkqhjwdv.js | `Use this MCP server` | project MCP server approval | Select, Dialog frame | Startup screens |
| chunk-kkqhjwdv.js | `Select any you wish to enable\.` | project MCP server approval | Dialog frame | Startup screens |
| chunk-kkwnj9ds.js | `Enter the path to the directory:` | add-directory input (/add-dir, /permissions) | TextInput | Slash-command panels |
| chunk-kkwnj9ds.js | `Add directory to workspace` | add-directory input (/add-dir, /permissions) | at hook, Select, onKeyDown, Dialog frame, KeyBox | Slash-command panels |
| chunk-krees34a.js | `Reading the cloud session's rules\\u2026` | slash command /permissions | at hook, exit keys, Tabs | Slash-command panels |
| chunk-krees34a.js | `Project settings \(local\)` | slash command /permissions | Select | Slash-command panels |
| chunk-krees34a.js | `Where should this rule be saved\?` | slash command /permissions | Select, Dialog frame | Slash-command panels |
| chunk-krees34a.js | `Permission rules are a tool name, optionally followed by a specifier i` | slash command /permissions | at hook, TextInput, Dialog frame | Slash-command panels |
| chunk-krees34a.js | `Commands recently denied by the auto mode classifier\.` | slash command /permissions | scoped bindings, Select, onKeyDown | Slash-command panels |
| chunk-krees34a.js | `Claude Code will no longer have access to files in this directory\.` | slash command /permissions | Confirm, Dialog frame | Slash-command panels |
| chunk-krees34a.js | `Workspace dialog dismissed` | slash command /permissions | Select | Slash-command panels |
| chunk-krees34a.js | `Built-in environment` | slash command /permissions | focus hook, Select, SearchBox | Slash-command panels |
| chunk-krees34a.js | `This editor writes only the modern keys \\u2014 move the rule to autoMo` | slash command /permissions | Select | Slash-command panels |
| chunk-krees34a.js | `This editor writes only the modern keys \\u2014 move the rule to autoMo` | slash command /permissions | at hook, Select×2 | Slash-command panels |
| chunk-krees34a.js | `Built-in rules` | slash command /permissions | at hook, Select | Slash-command panels |
| chunk-krees34a.js | `Delete auto mode rule\?` | slash command /permissions | at hook, Confirm | Slash-command panels |
| chunk-krees34a.js | `What kind of rule is this\?` | slash command /permissions | Select, Dialog frame | Slash-command panels |
| chunk-krees34a.js | `Deploys from this machine only reach the staging cluster\.` | slash command /permissions | at hook, TextInput, Dialog frame | Slash-command panels |
| chunk-krees34a.js | `This rule is configured by managed settings and cannot be modified\.` | slash command /permissions | at hook, Confirm | Slash-command panels |
| chunk-krees34a.js | `Claude Code won't ask before using allowed tools\.` | slash command /permissions | focus hook, Select, SearchBox | Slash-command panels |
| chunk-krees34a.js | `Permissions dialog dismissed` | slash command /permissions | at hook, Select, Confirm, onKeyDown, onPaste, exit keys, Dialog frame×2, Tabs, KeyBox | Slash-command panels |
| chunk-m246t60r.js | `function Ne\(\)\{if\(ma\(P,W\.` | CLI subcommand applyProjectSettingsHandler (staged change review) | at hook, exit keys, List | None: not part of the interactive session the brain runs |
| chunk-m246t60r.js | `Decide later` | CLI subcommand applyProjectSettingsHandler (staged change review) | Select | None: not part of the interactive session the brain runs |
| chunk-mbp8mjrp.js | `Copied to clipboard` | slash command /btw | onKeyDown, autoFocus | Slash-command panels |
| chunk-mkqxhzpz.js | `Find this session in the Code tab of the Claude app` | slash command /mobile | at hook, exit keys | Slash-command panels |
| chunk-mkqxhzpz.js | `Visit on your phone: ` | slash command /mobile | onKeyDown, KeyBox | Slash-command panels |
| chunk-msm1znd3.js | `Resetting your limits\\u2026 this usually takes a few seconds` | limit-reset / rate-limit option select (used by /rate-limit-options, /limit-reset) | Dialog frame | Slash-command panels |
| chunk-msm1znd3.js | `function g\(J\)\{let x=o\.fi` | limit-reset / rate-limit option select (used by /rate-limit-options, /limit-reset) | Select, Dialog frame | Slash-command panels |
| chunk-mvekzd4z.js | `Connect cloud sessions to GitHub\?` | slash command /web-setup | Confirm, Dialog frame×2 | Slash-command panels |
| chunk-mvx77wcz.js | `function W\(i\)\{let L=i\?\.p` | shared form Ly (fields + selects); each use listed | List | How a key travels (shared component: each use is listed under its own row) |
| chunk-mvx77wcz.js | `function z\(o\)\{let p=w\(75` | shared form Ly (fields + selects); each use listed | onKeyDown, onPaste, autoFocus | How a key travels (shared component: each use is listed under its own row) |
| chunk-my1jrmh8.js | `Save configuration` | plugin/marketplace configuration form (/plugin, /mcp, /skills) | FormDialog | Slash-command panels |
| chunk-mzzrt0by.js | `Add funds to continue with usage credits` | slash command /pro-trial-expired | Select, Dialog frame | Slash-command panels |
| chunk-n0z4vhva.js | `Code takes no children` | plugin-rendered Code element (plugin panes) | autoFocus | Above-prompt band, panes, footer selection and surveys |
| chunk-n3z8q4t9.js | `Talk to your codebase` | /powerup (chunk-xycvjzrb; also onboarding tour chunk-j79bfp5x) | on hook | Slash-command panels |
| chunk-n3z8q4t9.js | `Each power-up teaches one thing Claude Code can do that most people mi` | /powerup (chunk-xycvjzrb; also onboarding tour chunk-j79bfp5x) | Select | Slash-command panels |
| chunk-n3z8q4t9.js | `function Le\(p\)\{let l=w\(1` | /powerup (chunk-xycvjzrb; also onboarding tour chunk-j79bfp5x) | on hook | Slash-command panels |
| chunk-n5yz8vr1.js | `function \$G\(\{onClose:r,o` | $G detail-panel keys (Enter/space/left/x) for task detail views | on hook | Overlays drawn in place of the editor |
| chunk-pfjrt022.js | `Set up AWS Bedrock` | Bedrock provider setup (also /setup-bedrock) | FormDialog | Startup screens |
| chunk-pfjrt022.js | `Claude Code uses the standard AWS credential chain\. Pick the method yo` | Bedrock provider setup (also /setup-bedrock) | Select | Startup screens |
| chunk-pfjrt022.js | `API key is required` | Bedrock provider setup (also /setup-bedrock) | at hook, TextInput | Startup screens |
| chunk-pfjrt022.js | `These will be written to ` | Bedrock provider setup (also /setup-bedrock) | Confirm | Startup screens |
| chunk-pfjrt022.js | `Type a different name\\u2026` | Bedrock provider setup (also /setup-bedrock) | at hook, Select, TextInput | Startup screens |
| chunk-pfjrt022.js | `Region is required` | Bedrock provider setup (also /setup-bedrock) | at hook, TextInput | Startup screens |
| chunk-pfjrt022.js | `Authenticated as ` | Bedrock provider setup (also /setup-bedrock) | Select, Confirm | Startup screens |
| chunk-pjd1b1v6.js | `A design credential is already stored \\u2014 completing this flow repl` | slash command /design-login | onKeyDown, KeyBox | Slash-command panels |
| chunk-pjd1b1v6.js | `Starting design login\\u2026` | slash command /design-login | TextInput | Slash-command panels |
| chunk-pndhv0fp.js | `Jump to bottom` | fullscreen message list container ("Jump to bottom"): scroll keys only while focused | onKeyDown | Transcript view |
| chunk-prsva4j5.js | `function Xst\(\{onPaste:e,` | m_ focus hook and Xst paste/key helper | onPaste | How a key travels (shared component: each use is listed under its own row) |
| chunk-q2y4yyn0.js | `Looking for connected browsers\\u2026` | slash command /chrome | List | Slash-command panels |
| chunk-q2y4yyn0.js | `Claude in Chrome is turned off in /mcp\. Open /mcp, select claude-in-ch` | slash command /chrome | Select | Slash-command panels |
| chunk-q2y4yyn0.js | `Not enabled in this session` | slash command /chrome | Select, Dialog frame | Slash-command panels |
| chunk-q43kb6tz.js | `function Ao\(P\)\{let K=w\(8` | shared key container Ao (autoFocus default true); each use listed | onKeyDown×2, onPaste×2, autoFocus×2 | How a key travels (shared component: each use is listed under its own row) |
| chunk-qszsvrp7.js | `Browser actions still go through Claude's regular permission prompts b` | Claude in Chrome extension detected | Confirm, Dialog frame | Startup screens |
| chunk-r684ff3e.js | `Feedback drafts` | feedback drafts panel (/bug, /feedback, /exit) | onKeyDown, Dialog frame, KeyBox | Slash-command panels |
| chunk-r684ff3e.js | `This session` | feedback drafts panel (/bug, /feedback, /exit) | TextInput | Slash-command panels |
| chunk-r684ff3e.js | `This session` | feedback drafts panel (/bug, /feedback, /exit) | List | Slash-command panels |
| chunk-r718d5qt.js | `Please select how you'd like to continue` | consumer terms / privacy settings (also /privacy-settings) | Select, Dialog frame | Startup screens |
| chunk-r718d5qt.js | `Review and manage your privacy settings at` | consumer terms / privacy settings (also /privacy-settings) | onKeyDown, Dialog frame, KeyBox | Startup screens |
| chunk-r8gg2ks6.js | `No, keep them off` | slash command /cd | Confirm | Slash-command panels |
| chunk-r91cx3d7.js | `function i\(s,o\)\{if\(Zn\(\)\)` | slash command /tasks | TasksDialog, CloudTasksDialog | Slash-command panels |
| chunk-r9wfcn1z.js | `Open Claude Code in ` | TeleportRepoMismatchDialog (jvmshzp2 launcher, '--teleport') | Dialog frame, List | Startup screens |
| chunk-rgcbtyh6.js | `Autofix PR cancelled` | slash command /autofix-pr | Select, Dialog frame | Slash-command panels |
| chunk-rgcbtyh6.js | `Autofix PR cancelled` | slash command /autofix-pr | on hook, Dialog frame | Slash-command panels |
| chunk-t1x5k2p2.js | `You already have auto-mode entries \\u2014 add to them, or start fresh\?` | slash command /auto-mode-setup | Select×2, Dialog frame×2 | Slash-command panels |
| chunk-t1x5k2p2.js | `Optional reads \(Claude already reads this project\):` | slash command /auto-mode-setup | scoped bindings, Dialog frame, Form | Slash-command panels |
| chunk-t1x5k2p2.js | `Optional reads \(Claude already reads this project\):` | slash command /auto-mode-setup | Select (screen reader)×2, Dialog frame | Slash-command panels |
| chunk-t1x5k2p2.js | `The model call didn\\u2019t complete\. This is usually temporary \\u2014 ` | slash command /auto-mode-setup | at hook | Slash-command panels |
| chunk-t386ndqg.js | `Yes, I trust these settings` | managed settings approval | at hook, Confirm | Startup screens |
| chunk-t9jcg98t.js | `Feedback is disabled by your organization's policy\.` | /bug and /feedback (reached from chunk-5wpyqmam, chunk-7z794yae) | at hook | Slash-command panels |
| chunk-t9jcg98t.js | `Edit and press Enter to retry, or Esc to cancel` | /bug and /feedback (reached from chunk-5wpyqmam, chunk-7z794yae) | Select, onKeyDown, TextInput, Dialog frame, KeyBox | Slash-command panels |
| chunk-tbnp4j48.js | `function be\(i\)\{let d=w\(4` | Dialog frame 'be' (bE scope Confirmation, claimFocus; confirm:no=onCancel; Ctrl-C/Ctrl-D twice); each use listed | scoped bindings, exit keys | How a key travels (shared component: each use is listed under its own row) |
| chunk-tkn7e17a.js | `No changed files` | /diff dialog (chunk-qctfdyf4) | List | Slash-command panels |
| chunk-tkn7e17a.js | `Staged and new files` | /diff dialog (chunk-qctfdyf4) | overlay | Slash-command panels |
| chunk-tkn7e17a.js | `Staged and new files` | /diff dialog (chunk-qctfdyf4) | on hook×2, active context | Slash-command panels |
| chunk-tkn7e17a.js | `Diff dialog dismissed` | /diff dialog (chunk-qctfdyf4) | Dialog frame, Tabs | Slash-command panels |
| chunk-v1gvah25.js | `Typing is paused until the prompt opens` | cloud create checklist (remote launch) | on hook | Startup screens |
| chunk-vg4m2w1t.js | `Copy cancelled` | slash command /copy | at hook, exit keys | Slash-command panels |
| chunk-vg4m2w1t.js | `Always copy full response` | slash command /copy | onKeyDown, List | Slash-command panels |
| chunk-vkjeenf8.js | `AppStateProvider can not be nested within another AppStateProvider` | root key dispatcher (KeybindingSetup / onKeyDownCapture on the Global scope box) | onKeyDownCapture, keybindingScope | How a key travels (shared component: each use is listed under its own row) |
| chunk-vtbnqtw7.js | `Failed to resume the conversation\.` | ResumeConversation ('--resume' without an id / '--continue' picker) | pre-dispatch | Startup screens |
| chunk-w1cwhrb3.js | `Sign in to enroll this device for Remote Control\.` | slash command /remote-control | LoginPicker | Slash-command panels |
| chunk-w1cwhrb3.js | `Disconnect this session` | slash command /remote-control | overlay | Slash-command panels |
| chunk-w1cwhrb3.js | `A session's Project is fixed when it's created \\u2014 disconnect first` | slash command /remote-control | List | Slash-command panels |
| chunk-w4z50mrt.js | `Responding to ping` | Chrome native-messaging host | stdin listener×3, stdin data | None: not part of the interactive session the brain runs |
| chunk-wjtvbeqc.js | `Error: Selected environment not found` | slash command /remote-env | Dialog frame×3 | Slash-command panels |
| chunk-wjtvbeqc.js | `\(couldn't list environments: ` | slash command /remote-env | List | Slash-command panels |
| chunk-wxcct5md.js | `function K\(\$o\)\{if\(u\?\.typ` | shared form dialog UG (Select/Tabs/Settings contexts) used by provider setup; each use listed | at hook, on hook×2, Dialog frame | How a key travels (shared component: each use is listed under its own row) |
| chunk-wxcct5md.js | `function De\(Ye\)\{let F=w\(` | shared form dialog UG (Select/Tabs/Settings contexts) used by provider setup; each use listed | TextInput | How a key travels (shared component: each use is listed under its own row) |
| chunk-x2hxr175.js | `function fqe\(d,h=!0\)\{on\(` | shared search box Kb; each use listed | on hook | How a key travels (shared component: each use is listed under its own row) |
| chunk-x2hxr175.js | `function Kb\(\{query:d,pla` | shared search box Kb; each use listed | onKeyDown×2, onPaste×2, autoFocus×2, TextEntry | How a key travels (shared component: each use is listed under its own row) |
| chunk-xh32x4kb.js | `Yes, use my enabled plugins in cloud sessions` | slash command /cloud-plugins | Select, Dialog frame | Slash-command panels |
| chunk-xj6cj128.js | `Claude understands your codebase, makes edits with your permission, an` | slash command /help | List | Slash-command panels |
| chunk-xj6cj128.js | `Help dialog dismissed` | slash command /help | at hook, exit keys, Tabs | Slash-command panels |
| chunk-xkm2nhms.js | `An MCP server on this machine changed its tools` | device MCP reconsent (launchDeviceMcpReconsentDialog) | at hook | Startup screens |
| chunk-xkm2nhms.js | `This server has` | device MCP reconsent (launchDeviceMcpReconsentDialog) | Select, Dialog frame | Startup screens |
| chunk-xwmez4y2.js | `Scroll speed` | slash command /scroll-speed | onKeyDown, KeyBox | Slash-command panels |
| chunk-yqzv0rzr.js | `Starting new login following /usage-credits\. Exit with Ctrl-C to use e` | login method picker after /usage-credits | LoginPicker | Slash-command panels |
| chunk-yxd16f6e.js | `Loading session\\u2026` | session list / rename (/resume) | at hook×2 | Slash-command panels |
| chunk-yxd16f6e.js | `function ze\(B,K,Pt\)\{let ` | session list / rename (/resume) | Select, onKeyDown, autoFocus | Slash-command panels |
| chunk-yxd16f6e.js | `No conversations found\.` | session list / rename (/resume) | at hook, exit keys | Slash-command panels |
| chunk-yxd16f6e.js | `No conversations found\.` | session list / rename (/resume) | onKeyDown, onPaste, TextInput, SearchBox | Slash-command panels |
| chunk-yybr850r.js | `Directory ` | slash command /daemon | Select, Confirm, Dialog frame×2 | Slash-command panels |
| chunk-yybr850r.js | `Yes, trust and add server` | slash command /daemon | Confirm, Dialog frame, FormDialog | Slash-command panels |
| chunk-yybr850r.js | `Remote Control` | slash command /daemon | Dialog frame, Tabs | Slash-command panels |
| chunk-yybr850r.js | `Daemon service` | slash command /daemon | on hook | Slash-command panels |
| chunk-zcy1rm57.js | `This computer was removed from your devices` | device re-enroll (launchDeviceReenrollDialog) | at hook | Startup screens |
| chunk-zcy1rm57.js | `This computer was removed from your devices` | device re-enroll (launchDeviceReenrollDialog) | Select, Dialog frame | Startup screens |
