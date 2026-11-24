# Wake word on the phone page

Written 2026-11-23 from a conversation between Brandon and Claude. Ticket `promptctl-phone-app-9d9y.bw7`.

## The idea in one paragraph

The phone page listens for the wake word itself. A small detector runs inside the page, on the phone, on the same model files the Mac runs, so the phrase is the Mac's: "Hey Jarvis" unless `config.toml` names another, a word Brandon trained included. Until the detector hears the word, no microphone sample leaves the phone. When it hears it, the page sends the second it kept from before the detection, then the microphone as it comes, and hands opens a turn at the phone. The turn ends as a wake word turn ends at the desk, by end-of-turn detection on the Mac, and the page goes back to waiting for the word. A small button in the corner of the page turns this off and on. It is on by default.

## The picture

```
  the phone page                                          hands, on the Mac
 ┌───────────────────────────────────────────┐           ┌────────────────────────────────────┐
 │  microphone                               │           │                                    │
 │      │  through the browser's             │           │                                    │
 │      ▼  echo canceller                    │           │                                    │
 │  the page's tap                           │           │                                    │
 │      │                                    │           │                                    │
 │      ├──────► detector, in a Web Worker   │           │                                    │
 │      │        the Mac's three models      │           │                                    │
 │      │        on ONNX Runtime             │           │                                    │
 │      │              │ heard the word      │           │                                    │
 │      ▼              ▼                     │           │                                    │
 │  the last ───► turn state ◄── the button  │           │                                    │
 │  second             │                     │  "talk"   │                                    │
 │                     ▼                     │  channel  │                                    │
 │  one message every 20 ms: ────────────────┼──────────►│ Phone ─► the gate ─► the pipeline  │
 │    the microphone, inside a turn          │           │   │         ▲       (clip owner,   │
 │    a quiet marker, outside one            │           │   ▼         │        Whisper,      │
 │                                           │  opened,  │ the phone's driver   the brain)    │
 │  what the page says ◄─────────────────────┼─released, │ (Silero, Smart Turn)               │
 │                                           │  …        │                                    │
 │  speaker ◄────────────────────────────────┼───────────┤ hands' voice, on the audio track   │
 └───────────────────────────────────────────┘           └────────────────────────────────────┘
```

## A turn, step by step

1. Brandon opens the page and taps Connect, as today. The call comes up and hands is at the phone. The page loads the detector, tells hands that the phone listens for the wake word, and its line reads `Connected. Say "Hey Jarvis", or hold the button.` From here the page sends hands a quiet marker every 20 ms, and no microphone sample.
2. He says "Hey Jarvis, what's on the backlog." The detector scores the microphone every 80 ms. A score at or above the Mac's threshold is the word.
3. The page sends the last second it kept, then a `wake` message with the score, then the microphone as it comes.
4. On the Mac the gate opens a turn at the phone, by a trigger of its own, in the step that reads the `wake`. The tone that opens a turn plays on the phone, and the page is told which clip the turn is recorded in.
5. The phone's driver on the Mac hears the call from the wake on. Once he has spoken and stopped, Smart Turn judges the turn over and the gate sends it. The page is told its clip was released, and goes back to quiet markers.
6. Whisper hears the whole clip, wake word and question together, and the brain answers on the call's audio track.
7. If nothing is asked within eight seconds of the wake, the turn is thrown away, as at the desk, and the page is told so.

Holding the button works at every moment, as today, whether the wake word is on, off, loading or broken.

## Where this design departs from the words of the request

Each of these is Claude's, and Brandon's to overrule.

**The call stays up; only the microphone waits.** The request says the page opens the audio connection to the Mac only after the wake word. In this design the call is connected from the tap on Connect, and what waits for the wake word is the microphone's audio. The promise the request was after holds exactly: no audio leaves the phone before the wake word. The reasons for keeping the call up:

- hands speaks to the phone on the call: replies, tones, a task that finishes ten minutes later. hands is "at the phone" only while a call is up. With no call between turns, anything said after a turn ended would play at the desk, to an empty room.
- Opening a call takes a round trip to hands and two handshakes before the first sample can go. Paid on every turn, that is a pause after every "Hey Jarvis".
- It adds no privacy. The page decides what it sends in either design, and the phone's microphone reaches the Mac only as messages the page chooses to put on the data channel: the call carries no microphone track.

**The button's turns hold the microphone back too.** Today the page sends the microphone for the whole call and the Mac silences it outside a press. In this design the page has one rule, whichever way a turn opens: microphone samples leave the phone only inside a turn. With the wake word off, nothing leaves until the button is down. One rule is less to get wrong than two, and the corner button then only chooses whether the wake word may open a turn.

**The screen stays on.** A phone that sleeps stops listening, so the page holds a screen wake lock while the wake word is on and the call is up.

The design also overrules an earlier ticket. `hands-phone-app-bef.3pn`, an earlier agent's, has the Mac detect the wake word from the call's audio and says the trigger should be "one setting, not one per location". Neither holds under this request: a detector on the Mac needs the microphone before the word, and a phone page that is on by default cannot share a setting with a desk that is off by default. "The listening levels" below says what is left of that ticket.

## The four points

### The engine

**ONNX Runtime Web 1.31.1, its WebAssembly build, on one thread, in a Web Worker the page starts.** It runs the three files the Mac runs (`src/hands/voice/wake.py`, which `src/hands/voice/wakemodels.py` fetches them for): `melspectrogram.onnx`, `embedding_model.onnx`, and the word's own model. The Mac serves the page the very files it loads, so the model is the same, and the phrase is the same, for one of openWakeWord's words and for a word Brandon trained alike. The phrase the page shows and the threshold it holds a score to come from the Mac's own values (`wakeword.Word`, `wakeword.THRESHOLD`), so neither is written twice.

The scores are not the Mac's to the last digit, and are not meant to be: openWakeWord starts its buffer from random noise, and the phone's microphone is not the Mac's. Same model, same threshold, same phrase.

What runs around the models. openWakeWord is a Python library, so the page carries its streaming steps as a small module of its own: audio in 80 ms chunks at 16 kHz, a buffer of mel frames, a buffer of embeddings, the word's model over the last of them, the first few scores after a reset set to zero, and a reset after each hit so one saying wakes hands once. The page's audio runs at the phone's own rate, 48 kHz as a rule, and the module resamples it to 16 kHz itself: iOS cannot be relied on for a 16 kHz audio context (WebKit bug 251350).

Why one thread in a worker. openWakeWord's README says one core of a Raspberry Pi 3 runs 15 to 20 of its models in real time, so one model on one thread of a phone is well inside what the hardware does. Threads in WebAssembly need the page to be cross-origin isolated, which this page is not and need not become. The worker keeps the models off the page's main thread, so the 20 ms sender and the button never wait on one.

The runtime is two files: a script of 77 kB and a WebAssembly file of 14.8 MB, which the script fetches from beside itself. hands fetches both at the pinned version, holds each to a sha256 digest written in the code, as it does DTLN's files (`src/hands/voice/dtln.py`), and serves them itself. Nothing on the page loads from another site. The phone downloads the 14.8 MB once for each version of the runtime, and its browser keeps it: these two routes are served so a browser may keep them, under a name that carries the version, where the page and `hands.js` are served `no-store`.

That these three models run on this runtime in a browser is so far others' account (Deep Core Labs' port, under Sources), not something run here. The first build ticket's page test is the first time it runs here, in Chrome.

### The audio from before the detection

**The page keeps the last second of the microphone in a ring. On a hit it sends that second, then `wake`, then the microphone as it comes, in that order on the one ordered channel.**

On the Mac that second becomes the start of the clip by the mechanism the desk already has. While the desk listens, the clip owner keeps the last second it heard, and a clip opens with it (`src/hands/voice/clips.py`, `Recording`). The gate learns that the phone can listen between turns too, so the second the page sends ahead of `wake` is kept the same way, and the turn `wake` opens starts with it. No count of samples travels with the wake: the order on the channel is the boundary.

What the second is for:

- The words said right after the wake word. The detector is sure of the word a little after it ends, and "Hey Jarvis what time is it" has no gap to be sure in.
- The wake word itself, so Whisper hears the sentence whole, as it does at the desk.
- Cutting hands off. A turn the voice opened stops hands' reply once a person is heard in it, and the wake word in the kept second is that person.

Why a second: it is the Mac's own length (`clips.KEPT_SECONDS`), the Mac tells the page that length as a duration, and the clip owner keeps only the last second of what comes before a clip, so more would be thrown away.

The second is not there to cover a connection opening, since the call is already up.

The ring holds only audio from after the last turn closed, as the clip owner's recording does (it clears as each clip ends). The page empties its ring where a turn closes, so a word said within a second of a turn's end does not send that turn's last words again ahead of `wake`.

The phone's driver does not hear the kept second. It hears from the wake on, as the desk's driver hears afresh from the wake, because Smart Turn judges "Hey Jarvis" alone to be a finished sentence.

### Listening again after a turn

**The detector never stops hearing while the wake word is on. The page's one turn state decides what leaves the phone.**

```
                    the detector hears the word
  waiting ───────────────────────────────────────► the wake's turn
   ▲  ▲  │                                            │        │
   │  │  │  the button goes down                      │        │  hands says it is over:
   │  │  └──────────────► the button's turn ◄─────────┘        │  released, discarded
   │  │                          │       the button goes down  │  or unopened
   │  └──────────────────────────┘                             │
   │       the button comes up                                 │
   └───────────────────────────────────────────────────────────┘
```

- The detector hears the microphone for as long as the wake word is on and the call is up, turns included, and is reset after each hit, as the Mac's model is. A hit while a turn is open opens nothing.
- So there is no moment after a turn when the page is not listening. The word said as hands finishes speaking, or over hands' reply, is heard.
- The button's turn ends with the finger, as today. The wake's turn ends when hands says so, and the page never guesses. A call that ends closes whichever is open.
- hands says a wake's turn is over in one of three ways. Its clip was `released`: sent, by Smart Turn's verdict, by the turn's limit, by the button pressed over it, by the wake word being switched off, or by a hand at the desk taking hands there. Its clip was `discarded`: nothing was asked after the wake. Or the wake was `unopened`: it opened no clip, hands being at the desk.

**Every press and every wake is answered once, and every clip ends once.** The page numbers its presses, and a wake counts as one. hands answers each number exactly once: `opened`, naming the clip the turn is recorded in, or `unopened`, saying why there is none. Every clip hands opened ends in exactly one of `released` or `discarded`, naming it. The page closes the wake's turn on `unopened` for the wake's number, or on `released` or `discarded` for the clip that number opened, and on nothing else. This is what keeps the microphone from being left open by a message that never comes.

Two things in hands today fall short of that rule, and the build closes both. A clip thrown away is not told to the page, and does not say which clip it was. And a turn that opens and ends between two 20 ms frames is never recorded as a clip, so its press hears nothing back; a tap of the button shorter than a frame can do this today. The clip owner will say of each turn the gate counted either that it opened a clip or that it never heard it, and a press whose turn was never heard is answered `unopened`.

### The listening levels

**The phone has a listening level of its own, with the desk's names: `off` and `wake word`.** The corner button moves it between the two.

What matches the desk's wake word level (`src/hands/voice/desk.py`):

| | at the desk | at the phone |
|---|---|---|
| the word and its model | `config.toml`'s | the same files, served by the Mac |
| the threshold | `wakeword.THRESHOLD` | the same value, served by the Mac |
| what the turn starts with | the second before the wake | the second before the wake |
| nothing asked | thrown away after `ASK_SECONDS`, 8 s | the same, by the same code |
| the end of the turn | Smart Turn's verdict, or the turn's limit | the same, by the same code |
| the hand during the turn | the talk key sends the turn and opens its own | the page's button does the same |
| the level switched off during the turn | the turn is sent | the turn is sent |
| the word said over hands' reply | cuts the reply off | cuts the reply off |
| the tones | open, sent, thrown away | the same, played on the phone |

What differs, and why:

- **Where the word is heard.** At the desk the Mac hears the room for as long as the level is on. At the phone the phone hears it, and the Mac hears nothing until the word.
- **The default.** Off at the desk, on at the phone. That is why the level cannot be one setting for both places: a phone page that is on by default must not open the Mac's microphone to the room.
- **Who owns it.** The page's stored choice. The page tells hands the phone listens once the wake word is on and the detector is ready, and tells it each change after. hands holds the level for as long as the call lasts, and a new call starts at `off`.

**A wake at the phone opens a turn only while hands is at the phone.** This is the gate's rule today for every voice: "a voice the desk hears, which can be anyone's or nobody's, opens turns only where hands is" (`docs/architecture.md`). A call that comes takes hands to the phone, so the plain case works with no hand at all. After a turn at the desk's talk key, hands is at the desk, and a wake at the phone opens nothing: the page says hands is at the desk and that holding the button brings it back. The rule earns its place when the phone and the desk are in one room. One "Hey Jarvis" then opens one turn, where hands is, and a word the phone mishears cannot take a reply off the desk.

**A press over a wake's turn sends it and opens the button's own**, as the talk key does over a voice turn at the desk. What was said before the press and what is said under it are two clips. The cost is that a sentence the button interrupts is heard in two pieces, as it is at the desk today.

Engaged conversation at the phone is not this design. It needs the Mac to hear the call between turns, which is the opposite promise. It stays with `hands-phone-app-bef.3pn`, which builds on this design: the phone's driver built here is what it will hear with.

## The button, and what the page says

The page stays one page. It gains one small button in a corner, a toggle that reads `Phone wake word` and shows whether it is on. It is on unless the user turned it off; the choice is kept in the browser's storage, as the transcript's is. Off, the detector does not run and the screen is left to sleep.

The options page already has a control headed `Listening`, which sets the desk's level. Its heading becomes `Listening at the desk`. Each of the two controls then names the place it sets, and neither reads as the other's setting.

| | the page's line |
|---|---|
| wake word on and ready | `Connected. Say "Hey Jarvis", or hold the button.` |
| wake word off, or still loading | `Connected. Hold the button and speak.` |
| the detector could not load | the same line, and why the wake word is not listening |
| a wake with hands at the desk | `hands is at the desk. Hold the button to bring it here.` |
| a wake with nothing asked after it | `Heard nothing after "Hey Jarvis".` |
| a press whose turn hands never heard, or threw away | `Heard nothing from that press.` |

- Connect is still the one tap. A browser gives a page the microphone and sound only after a tap, so the wake word listens from Connect on, not from the page opening. When the call drops and the page calls again by itself, as it does today, it listens again without a tap.
- The corner button works before Connect, and the page fetches the detector only after Connect, with the wake word on. So with the wake word off the page is today's page, and if the detector ever takes the page down, off before Connect is the way back.
- The button always works. The detector loading, failing or switched off changes nothing about holding to talk.
- A failed detector is said in words on the page, and told to hands, so it is in hands' events too.
- The page asks the browser to keep the screen on while the wake word is on and the call is up, and asks again when the page comes back to the front.
- A page in the background hears nothing: iOS takes the microphone from a web page that is not in front. Listening with the screen locked is the native app's (`promptctl-iphone-app-dan0.0xn`).

## Between the page and hands

All of it is on the call's data channel, "talk", which delivers in order.

From the page:

| message | when | |
|---|---|---|
| audio, 16-bit at the call's rate | every 20 ms inside a turn; and the kept second, just ahead of a `wake` | today's message, no longer sent outside a turn |
| `quiet`, with a count of samples | every 20 ms outside a turn | new |
| `press`, `release` | the button | as today |
| `wake`, with the detector's score | the detector heard the word and no turn is open | new |
| `level`, `wake word` or `off` | once the wake word is on and the detector is ready, and on each change after; a call starts at `off` | new |
| `detector`, loaded or failed and why, and the browser the page is in | once for each call that loads the detector | new |

From hands:

| message | when | |
|---|---|---|
| `opened`, naming the press and the clip | a press or a wake opened a turn, and this is its clip | today's message; a wake counts as a press |
| `unopened`, naming the press and why | a press or a wake opened no clip: hands is at the desk, or its turn was over before hands heard any of it | new |
| `released`, naming the clip | the turn was sent | as today |
| `discarded`, naming the clip | the turn was thrown away: nothing was asked | new |
| milestones and transcript lines | as today | |

**Why a quiet marker, and not silence on the wire.** At the phone, the page's audio messages are the pipeline's clock. Each one becomes a frame, the clip owner closes a clip when the next frame arrives, and hands hangs up on a page that sends nothing for three seconds (`src/hands/voice/phone.py`). A page that sent nothing between turns would leave every clip open and lose its call. So outside a turn the page sends, on the same 20 ms, a marker that says only how many samples it held back, and the Mac makes that much silence and treats it as audio. One message every 20 ms either way: the turn only decides what is in it. Everything downstream of `Phone` is untouched, and what the phone sends between turns falls from the whole microphone to a few bytes a message.

A turn's edge ends the message being gathered, once the tap's mark has come back for it, so every block posted before the button moved has been gathered. What the page had gathered before a press goes as a count, and what it gathered before a release goes as audio, so no sample from outside a turn rides in a message from inside one.

A page's message is never exactly 20 ms: it is cut at the first 128-sample block that reaches 20 ms, so at 48 kHz it holds 1024 samples, and a turn's edge only cuts one sooner. So the limit on a count is not 20 ms, and the Mac does not copy the page's cut rule to find it. It is a limit of the Mac's own, 100 ms at the call's rate, applied to a message's audio and a count alike, wide enough that a page cutting its messages another way is still read. It is not the span `Phone` takes a page for gone: a page that stalls sends many small counts, never one large one, so a limit of that span would let one bad message make about 150 times real time of silence. Audio or a count over the limit, and a count of no samples, is refused, logged as a message hands cannot read, and does not count as the page being alive. The limit is on one message, not on how many come: a page that sends messages faster than its microphone makes them is not what it guards against.

## On the Mac

- **A trigger of its own.** `Trigger` gains a value for a wake heard on the phone, beside `phone button`. Its place is the phone, it is not made by hand, and its turn cuts hands off once a person is heard in it. It is a new trigger and not a flag on the desk's `wake word`, because the two differ in place, and place is what the gate reads.
- **The gate holds listening for each place.** Today it holds the desk's alone. `listen` and `deafen` move the listening of the place their trigger belongs to, and the key rests at `listening` where the place hands is at listens. The gate's rule for who may open a turn does not change: it already takes this trigger at the phone and refuses it at the desk.
- **The gate is the one authority on the phone's turn.** It holds whether a turn is open and which one (`key`, `opened`, `turn`). `Phone` reads each message off the channel and moves the gate in the step that reads it: `wake`, `press`, `release` and `level`, as a press does today, so the audio sent after a `wake` is heard inside its turn and the second sent before it is kept. The phone's driver keeps a life for the turn the voice opened, and that life follows the gate: a wake enters it only when the gate took the wake, and every move that ends the gate's turn ends the life with it, whether the gate refused the wake because hands is at the desk, a hand at the desk took hands there, the call went, or the level went off. Every ear event and every verdict names the gate's turn of the audio it came from (`KeyedAudio.turn` carries it), as the deadlines already do, so an event for a turn that is over steps nothing of the next.
- **`Phone` reads the new messages.** `quiet` becomes silence. `wake` is counted as a press and opens a turn, or is answered `unopened` at once. `level` moves the phone's listening. A `wake` that arrives while a turn is open, or while the phone's level is off, cannot come from a correct page, since the page's own turn state covers the Mac's and `level` rides the same ordered channel. It opens nothing, is answered `unopened`, and is logged as the page's mistake.
- **One life for a voice turn.** The rules for a turn the voice opened are in two files today, written around the desk. One is the step (`desk.py`): the pause after the wake word that is no end, the eight seconds for what is asked, the verdict, the limit. The other is the work around the models that goes with it (`engaged.py`): hearing afresh from the wake, what Smart Turn is fed, judging, clearing as a turn ends, setting the deadlines. Both move to one place that the desk and the phone run, each on its own ears, so the phone's turn ends by the same rules and the same numbers. The talk key and the taps stay the desk's own.
- **A driver for the phone.** It runs that life for the phone. It is handed each wake's turn already open, with the call's audio from the wake on in order, and owns only how the turn ends. It has ears of its own, a second Silero and Smart Turn (`engaged.Models`), because each pair holds the state of one stream. They load as hands starts, as the desk's do, so a model that cannot load is the start's failure and never a wake that fails as it is heard. Its moves go through the gate and onto the phone's cues, so its tones play on the phone.
- **The clip owner accounts for every turn.** It tells of each clip it opens, as today; of each clip thrown away, by its number; and of each turn the gate counted that it never heard. `Phone` turns those into `opened`, `discarded` and `unopened`.
- **The listening tools speak for the place hands is at.** `listening_level` and `set_listening_level` (`src/hands/voice/tools.py`) read the desk's level and answer in the desk's words, how to talk at the desk included. At the phone, `listening_level` says the phone's level and that the page's `Phone wake word` button sets it, and `set_listening_level` says the same and changes nothing, since the page is the one owner of the phone's level. At the desk they are as they are.
- **Each wake is an event**, with the score, the threshold, whether hands was speaking, and whether it opened a turn. The detector's load, or its failure and why, is an event too, with the browser the page said it is in, so hands' events tell an iPhone's Safari from a desktop's Chrome.
- **Serving.** hands serves the page the detector's script, the runtime's two files, the three models, and the wake word's settings: the phrase, the threshold, the length of the kept second. The settings and the models are under the phone's key, since a trained model is the user's own. The runtime is public and open, as the page itself is. Asking for the settings is what makes hands fetch any file it does not yet have, so a failed fetch is said to the page that asked, in hands' words.

## What can go wrong

- **The detector cannot load.** The page says so and why, hands records it, and the button works. Nothing else changes.
- **iOS kills the page for memory.** This is the one real unknown. What is known:
  - WebKit bug 304810, "High memory usage in WASM compilation thread", is open against Safari on iOS 26. A commenter there says it makes "many AI models loaded through ONNX runtime" fail on iOS 26, and points at ONNX Runtime's issue 26827, which could not be opened from here.
  - WebKit's engineer answers on the bug that they have been addressing it, and the ONNX issue, by changes to the compiler's allocators. The seven bugs it depends on are all marked fixed, and a duplicate was reported fixed in the iOS 26.5 beta. The bug itself names no release that fixes it.
  - In September 2026 a commenter measured iOS 26.6.2 still killing a page, four times in four. That page ran SQLite, not ONNX Runtime, in a WebAssembly build made with Emscripten's Asyncify. The same code built without Asyncify was not killed in three tries. WebKit asked for that to be filed as a new bug.
  - Checked here: the script of the runtime build this design uses has no Asyncify code in it (the word does not appear in the file), and ONNX Runtime's documentation lists this build, WebAssembly on the CPU, as supported in Safari on iOS. The 14.8 MB WebAssembly file itself was not opened.
  - Not known: whether Safari keeps this page up on Brandon's iPhone. Nothing here can answer that, and the last ticket is that check. If Safari cannot keep the page up, that comes back to this document; it is not for the build to work around. Until then the corner button, off before Connect, keeps the detector from loading at all.
- **hands' own voice says the phrase.** On the phone's speaker, hands saying "Hey Jarvis" in a reply reaches the microphone through the browser's echo canceller, which may not take all of it out. Such a wake shows in its event as heard while hands was speaking. Earbuds remove it.
- **A word misheard.** As at the desk: the turn is thrown away after eight seconds with nothing asked, and each wake is an event with its score.
- **Battery.** A screen that stays on and a model that runs cost battery. The corner button is how to stop both.
- **The licence of the pretrained words.** openWakeWord's code is Apache 2.0; its pretrained models are CC BY-NC-SA 4.0, non-commercial. That is the Mac's position today, and serving the same files to the phone does not change it.

## Why this and not the alternatives

- **Open the call only at the wake.** Answered above: hands could not speak to the phone between turns, every turn would pay for a call's setup, and nothing more would stay on the phone.
- **Detect on the Mac, from the call's audio** (`hands-phone-app-bef.3pn`). The microphone would leave the phone for the whole call, which is what this design is for not doing.
- **Send nothing at all between turns.** The pipeline's clock at the phone is the page's messages. Sending nothing means a second clock on the Mac, and new rules for the clip owner, the liveness watch and the release's ordering.
- **Another engine.** openWakeWord also ships TensorFlow Lite models, and there are ports of it to the browser by others. Either puts a second model format or someone else's unpinned code beside the Mac's ONNX files. The runtime chosen is the browser build of the one the Mac already uses.
- **The graphics card.** ONNX Runtime's documentation lists neither WebGPU nor WebNN as supported in Safari on iOS, and the models are small enough not to need them.
- **Score the kept second again on the Mac.** One detector decides for each place. A second opinion that disagrees has nobody to settle it.
- **Let a wake at the phone bring hands to the phone.** With the desk at its wake word level in the same room, one saying would open a turn at each place.
- **Let a press join the wake's turn**, so an interrupted sentence stays one clip. The turn's life is one piece of code for the desk and the phone, and at the desk the talk key sends a voice turn and opens its own. A phone that joined would be a second rule in that code, and a clip with two openers.

## How it is tested

No rig, and nothing measured.

- **The detector's module** runs in QuickJS on fake models, beside `tests/test_handsjs.py`: the chunking, the buffers, the reset, the threshold, the resampling.
- **The page** runs in Chrome under Playwright with the real runtime and models (`tests/test_pages.py`). Chrome plays a WAV as its microphone, speech synthesized as the other wake word tests synthesize it (`tests/speech.py`). The test that matters: hands is sent no microphone sample before the word, and a turn opens after it.
- **The Mac's side** runs in process, on messages fed in as the page sends them (`tests/test_phone.py`, and the driver's tests beside `tests/test_engaged.py`). A turn that opens and ends with no frame between is made by sending its messages back to back, never by timing.
- **Brandon's iPhone** is the one check no test here can make, and is its own ticket.

## Where the build lands

| file | what changes |
|---|---|
| `src/hands/voice/phone.html` | the corner button, the turn state, the ring, the one sending rule, the answers it reads, the page's lines, the wake lock |
| a new script beside `src/hands/voice/hands.js` | the detector: the worker, the resampler, openWakeWord's streaming steps |
| `src/hands/voice/phonepage.py` | the routes for the script, the runtime, the models and the settings |
| a new module beside `src/hands/voice/dtln.py` | the runtime's two files, fetched and held to their digests |
| `src/hands/voice/phone.py` | the new messages in and out; the phone's turn stepped as each message is read |
| `src/hands/voice/clips.py`, `trigger.py` | the new trigger: its place, that no hand made it, how its turn cuts hands off |
| `src/hands/voice/tools.py` | the listening tools report and refuse in the phone's terms while hands is at the phone |
| `src/hands/voice/ptt.py` | listening for each place |
| `src/hands/voice/desk.py`, `engaged.py`, and a new module they share | the voice turn's life in one place, which the desk runs |
| a new module for the phone, as `atdesk.py` is for the desk | the phone's driver and its ears |
| `src/hands/voice/clipowner.py`, `turnstop.py`, `latency.py`, `milestone.py` | a clip thrown away names itself, a turn never heard is told, and the page is told both; `src/hands/brain/stage.py` reads the thrown-away clip's frame and keeps its meaning |
| `src/hands/voice/gallery.html` | the desk's control names the desk |
| `src/hands/daemon/run.py` | the phone's driver started with hands |
| `docs/architecture.md` ("The audio side"), `README.md`, `docs/testing.md` | the trigger table, listening for each place, the phone's audio, the wake word on the phone, the page test's microphone |

## The tickets

Seven tickets under epic `promptctl-phone-app-9d9y`, in one lane, `phone-wake`, behind this design's own ticket. Each is built and merged before the next starts.

1. `promptctl-phone-app-9d9y.ycy`: the detector runs the Mac's models in a browser, on a runtime hands serves. "The engine".
2. `promptctl-phone-app-9d9y.671`: the page sends its microphone only inside a turn, and a count of what it held back outside one. "Between the page and hands".
3. `promptctl-phone-app-9d9y.ou2`: every press is answered once, `opened` or `unopened`, and every clip ends once, `released` or `discarded`. "Listening again after a turn".
4. `promptctl-phone-app-9d9y.nzq`: the life of a voice turn is in one place, which the desk runs on its own ears. "One life for a voice turn".
5. `promptctl-phone-app-9d9y.vlf`: a wake heard on the phone opens a turn at the phone, which ends as a wake word turn ends at the desk. "On the Mac", "The audio from before the detection", "The listening levels".
6. `promptctl-phone-app-9d9y.z1k`: the page listens for the wake word, on by default, with the corner button. "The button, and what the page says".
7. `promptctl-phone-app-9d9y.mj9`: a call from Brandon's iPhone shows whether Safari keeps the page up. It carries the `external` label, since its event is outside the repository, so nothing waits on it.

Between them they build every row of "Where the build lands". After this design is accepted, `hands-phone-app-bef.3pn` is rewritten to what is left of it, engaged conversation at the phone.

## Sources

- [onnxruntime-web 1.31.1, its files on jsDelivr](https://data.jsdelivr.com/v1/packages/npm/onnxruntime-web@1.31.1?structure=flat), and [the bundled script](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.wasm.bundle.min.mjs), read to see that it carries its own glue and fetches the one WebAssembly file from beside itself
- ONNX Runtime's documentation: [deploying on the web](https://onnxruntime.ai/docs/tutorials/web/deploy.html), which says which WebAssembly file each build uses; [getting started](https://onnxruntime.ai/docs/get-started/with-javascript/web.html), whose table says what runs in Safari on iOS; [flags and session options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html); [the WebAssembly flags](https://onnxruntime.ai/docs/api/js/interfaces/Env.WebAssemblyFlags.html)
- [openWakeWord's README](https://github.com/dscripka/openWakeWord/blob/main/README.md), for its speed, its 80 ms frames and its licences, and [its release v0.5.1](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1), the Mac's files
- [WebKit bug 304810](https://bugs.webkit.org/show_bug.cgi?id=304810), read whole, comments and dependencies, and [onnxruntime issue 26827](https://github.com/microsoft/onnxruntime/issues/26827), named there and not opened from here
- [WebKit bug 251350](https://bugs.webkit.org/show_bug.cgi?id=251350), the 16 kHz audio context on iOS
- [Deep Core Labs, openWakeWord on the web](https://deepcorelabs.com/open-wake-word-on-the-web), one port's account of the same three models in a browser
- Chrome's fake microphone from a file: [the media switches](https://chromium.googlesource.com/chromium/src/+/refs/tags/112.0.5615.26/media/base/media_switches.cc)
