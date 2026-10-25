import { h, intro, notes, box, tokSpan, presets, debounce, errorLine } from "../ui.js";
import { post } from "../api.js";

const EXAMPLES = [
  { name: "A sentence", text: "The owl of Minerva spreads its wings only with the falling of the dusk." },
  { name: "Long words", text: "Unbelievably, the antidisestablishmentarianists were incomprehensibly uncharacteristic." },
  { name: "strawberry", text: "How many r's are in the word strawberry?" },
  { name: "Numbers", text: "In 1781 Kant published 856 pages; 2 + 2 = 4, but 12345 × 6789 = 83810205." },
  { name: "Names", text: "Wittgenstein, Anscombe, Merleau-Ponty, Zhuangzi, Nāgārjuna and Ibn Rushd." },
  { name: "Other languages", text: "Ich denke, also bin ich. Je pense, donc je suis. 我思故我在. Cogito, ergo sum." },
  { name: "Code", text: "def is_even(n):\n    return n % 2 == 0" },
  { name: "Emoji", text: "Socrates 🧔‍♂️ drank the hemlock 🍵☠️." },
];

export default {
  id: "tokens",
  title: "Tokens",
  mount(main) {
    const input = h("textarea", { rows: 3, "aria-label": "Text to split into tokens" }, EXAMPLES[0].text);
    const chips = h("div", { class: "tokens", "aria-live": "polite" });
    const ids = h("div", { class: "ids" });
    const nTok = h("b", {}, "–"), nWord = h("b", {}, "–"), nChar = h("b", {}, "–");
    const msg = h("div");

    const run = debounce(async () => {
      const text = input.value;
      nChar.textContent = text.length;
      nWord.textContent = (text.match(/\S+/g) || []).length;
      if (!text) { chips.replaceChildren(); ids.replaceChildren(); nTok.textContent = 0; return; }
      try {
        const { tokens } = await post("/api/tokenize", { text });
        msg.replaceChildren();
        nTok.textContent = tokens.length;
        chips.replaceChildren(...tokens.map((t) => tokSpan(t.text, { class: `tok${t.special ? " special" : ""}`, title: `token #${t.id}` })));
        ids.replaceChildren(...tokens.map((t) => h("span", {}, t.id)));
      } catch (e) { errorLine(msg, e); }
    }, 150);
    input.addEventListener("input", run);

    main.append(
      intro("Step 0 · Tokens",
        "The model never sees letters or words",
        "Before anything else, text is cut into tokens: common words, pieces of words, single characters. Each token is replaced by a number from a fixed list of about 150,000. From here on, the model only ever deals with these numbers."),
      h("div", { class: "stack" },
        box("Type anything", {
          right: presets(EXAMPLES, (p) => { input.value = p.text; run(); }, { active: EXAMPLES[0] }),
          body: [input, msg],
        }),
        box("What the model receives", {
          body: [
            h("div", { class: "stats" },
              h("div", { class: "stat" }, nChar, h("span", {}, "characters")),
              h("div", { class: "stat" }, nWord, h("span", {}, "words")),
              h("div", { class: "stat" }, nTok, h("span", {}, "tokens"))),
            h("span", { class: "label" }, "As tokens · a dot marks a space"),
            chips,
            h("span", { class: "label" }, "As numbers · this is all the model gets"),
            ids,
          ],
        })),
      notes({
        math: [
          "The token list (the <em>vocabulary</em>) is built before training by repeatedly merging the most frequent pair of adjacent symbols in a huge text collection (“byte-pair encoding”). Frequent words become one token; rare words are spelled out from pieces.",
          { f: "“ Wittgenstein” → [ ' Witt', 'gen', 'stein' ] → [ 82176, 4370, 12429 ]" },
          "The numbers are arbitrary labels: token 82176 is not “bigger” than token 4370. Their meaning comes only from what the model learns about each one.",
        ],
        discuss: [
          "For this model, “ strawberry” is a single token, number 72600: it never sees the letters. Why might it then struggle to count the r's, yet write a fluent essay about strawberries?",
          "Common English words get one token; a Sanskrit or Yoruba word may need five. What does that mean for cost, speed and quality in different languages?",
          "Is a token a symbol in the philosopher's sense? What, if anything, does it refer to?",
        ],
      }),
    );
    run();
  },
};
