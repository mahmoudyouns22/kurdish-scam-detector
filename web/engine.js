/* Browser-native inference for kurdish-scam-detector.
 *
 * A faithful port of src/normalize.py, src/signals.py and the scikit-learn
 * feature union + logistic regression. Everything runs on the device: no
 * server, no request, no message ever leaves the browser. For a tool that asks
 * people to paste an SMS containing their bank details, that is the point.
 *
 * web/verify_parity.py checks this file against the Python pipeline on every
 * corpus row. If they disagree, this file is wrong.
 */
(function (global) {
  "use strict";

  // ---- Python -> JavaScript regex translation ------------------------------
  //
  // Python's `re` is Unicode-aware where JavaScript's is not, and the
  // difference is not cosmetic on Arabic-script text:
  //
  //   \d  Python matches ٠-٩ and ۰-۹; JS matches ASCII only. A raw-IP URL
  //       written as http://١٨٥.٢٠٣.١١.٤٢ is caught by one and missed by the
  //       other.
  //   \b  Python treats Arabic letters as word characters, so "500000 دینار،"
  //       has a boundary after دینار. JS does not, so the money pattern simply
  //       failed to match and the amount fell through to __num__.
  //
  // Both were found by web/verify_parity.py, not by reading the code.

  const WORD_CHAR = "[\\p{L}\\p{N}_]";
  // Exact emulation of a Unicode word boundary: a word/non-word transition in
  // either direction. Works in leading and trailing position alike.
  const UNI_WB =
    "(?:(?<=" + WORD_CHAR + ")(?!" + WORD_CHAR + ")|(?<!" + WORD_CHAR + ")(?=" + WORD_CHAR + "))";

  // Escapes JavaScript's `u` mode accepts. Anything else keeps the character
  // but loses the backslash: Python's r"\"" is a literal quote, but \" is an
  // invalid identity escape under `u` and throws at compile time.
  const KEEP_ESCAPE = new Set("^$\\.*+?()[]{}|/-dDsSwWbBnrtfv0123456789xupPkAZzGQE".split(""));

  function pyRegexToJs(src) {
    let out = "";
    let inClass = false;
    for (let i = 0; i < src.length; i++) {
      const c = src[i];
      if (c === "\\") {
        const n = src[i + 1];
        if (n === undefined) { out += "\\\\"; break; }
        if (n === "d") { out += inClass ? "\\p{Nd}" : "[\\p{Nd}]"; i++; continue; }
        if (n === "b" && !inClass) { out += UNI_WB; i++; continue; }
        out += KEEP_ESCAPE.has(n) ? "\\" + n : n;
        i++;
        continue;
      }
      if (c === "[" && !inClass) { inClass = true; out += c; continue; }
      if (c === "]" && inClass) { inClass = false; out += c; continue; }
      out += c;
    }
    return out;
  }

  function compile(src, flags) {
    return new RegExp(pyRegexToJs(src), flags + "u");
  }

  // ---- normalisation (port of normalize.py) --------------------------------

  function buildNormalizer(cfg) {
    const invisible = new Set(cfg.invisible);
    const tatweel = cfg.tatweel;
    const fold = cfg.letter_fold;
    const diacritics = new RegExp(cfg.diacritics_re, "u");

    // Returns {text, offsets} where offsets[i] is the index in the raw string
    // that produced text[i]. The demo highlights spans in the user's own text,
    // so this map has to be exact.
    return function normalizeTraced(raw) {
      const out = [];
      const offsets = [];
      const chars = Array.from(raw);
      // Array.from splits by code point; offsets must be UTF-16 indices to
      // match String.slice in the highlighter.
      let unitIdx = 0;
      for (let i = 0; i < chars.length; i++) {
        const ch = chars[i];
        const here = unitIdx;
        unitIdx += ch.length;
        if (invisible.has(ch) || ch === tatweel) continue;
        if (diacritics.test(ch)) continue;

        let folded = ch.normalize("NFKC");
        let mapped = "";
        for (const c of folded) mapped += fold[c] !== undefined ? fold[c] : c;
        mapped = mapped.toLowerCase();

        for (const c of mapped) {
          const isDigit = c >= "0" && c <= "9";
          const n = out.length;
          if (!isDigit && n >= 2 && out[n - 1] === c && out[n - 2] === c) continue;
          out.push(c);
          offsets.push(here);
        }
      }

      const res = [];
      const resOff = [];
      let prevSpace = false;
      for (let i = 0; i < out.length; i++) {
        const c = out[i];
        if (/\s/.test(c)) {
          if (prevSpace || res.length === 0) continue;
          res.push(" ");
          resOff.push(offsets[i]);
          prevSpace = true;
        } else {
          res.push(c);
          resOff.push(offsets[i]);
          prevSpace = false;
        }
      }
      while (res.length && res[res.length - 1] === " ") {
        res.pop();
        resOff.pop();
      }
      return { text: res.join(""), offsets: resOff };
    };
  }

  // ---- analyzers (port of sklearn's char_wb and word n-grams) --------------

  function charWbNgrams(doc, minN, maxN) {
    // sklearn collapses runs of 2+ whitespace, then splits.
    const text = doc.replace(/\s\s+/g, " ");
    const grams = [];
    for (const wRaw of text.split(/\s+/)) {
      if (!wRaw) continue;
      const w = " " + wRaw + " ";
      const wLen = w.length;
      for (let n = minN; n <= maxN; n++) {
        let offset = 0;
        grams.push(w.slice(offset, offset + n));
        while (offset + n < wLen) {
          offset += 1;
          grams.push(w.slice(offset, offset + n));
        }
        // A word shorter than n is counted once, and longer n are skipped.
        if (offset === 0) break;
      }
    }
    return grams;
  }

  // Python's (?u)\b\w\w+\b is a maximal run of 2+ unicode word characters.
  const WORD_RE = /[\p{L}\p{N}_]{2,}/gu;

  function wordNgrams(doc, minN, maxN) {
    const tokens = doc.match(WORD_RE) || [];
    let out;
    let lo = minN;
    if (maxN !== 1) {
      if (minN === 1) {
        out = tokens.slice();
        lo = 2;
      } else {
        out = [];
      }
      const nTok = tokens.length;
      for (let n = lo; n < Math.min(maxN + 1, nTok + 1); n++) {
        for (let i = 0; i <= nTok - n; i++) {
          out.push(tokens.slice(i, i + n).join(" "));
        }
      }
      return out;
    }
    return tokens;
  }

  // ---- signals (port of signals.py) ---------------------------------------

  function buildSignals(cfg, rx) {
    const urlRe = compile(rx.url, "gi");
    const phoneRe = compile(rx.phone, "g");
    const moneyRe = compile(rx.money, "gi");
    const ipv4Re = compile("https?://(?:\\d{1,3}\\.){3}\\d{1,3}", "");
    // Escapes, not literal characters. Written literally, this range is the
    // first thing to die if the file is ever served or saved as anything but
    // UTF-8 -- the class becomes "range out of order", throws at parse time,
    // and the whole engine fails to define itself.
    const arabicRe = /[\u0600-\u06FF\u0750-\u077F]/g;
    const latinRe = /[a-z]/gi;

    const legit = new Set(cfg.legit_domains);
    const shorteners = new Set(cfg.shorteners);
    const tlds = cfg.suspicious_tlds;
    const brandTokens = cfg.brand_tokens;

    function domainsOf(text) {
      const out = [];
      urlRe.lastIndex = 0;
      let m;
      while ((m = urlRe.exec(text)) !== null) {
        if (m[0].length === 0) { urlRe.lastIndex++; continue; }
        let host = m[0].replace(/^https?:\/\//i, "").split("/")[0];
        host = host.toLowerCase().split(":")[0];
        if (host.startsWith("www.")) host = host.slice(4);
        out.push(host);
      }
      return out;
    }

    function allMatches(re, text) {
      const out = [];
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(text)) !== null) {
        if (m[0].length === 0) { re.lastIndex++; continue; }
        out.push({ start: m.index, end: m.index + m[0].length, text: m[0] });
      }
      return out;
    }

    function lexHits(norm, offsets, lexicon, label, spans) {
      let hits = 0;
      for (const phrase of lexicon) {
        if (!phrase) continue;
        let start = 0;
        for (;;) {
          const i = norm.indexOf(phrase, start);
          if (i < 0) break;
          hits += 1;
          if (offsets.length) {
            const lo = offsets[i];
            const hi = offsets[Math.min(i + phrase.length - 1, offsets.length - 1)] + 1;
            spans.push([lo, hi, label]);
          }
          start = i + phrase.length;
        }
      }
      return hits;
    }

    return function analyse(text, normalizeTraced) {
      const { text: norm, offsets } = normalizeTraced(text);
      const f = {};
      const spans = [];

      const urls = allMatches(urlRe, text);
      const domains = domainsOf(text);
      f.url_present = urls.length ? 1 : 0;
      f.url_count = urls.length;
      for (const u of urls) spans.push([u.start, u.end, "link"]);

      f.url_shortener = domains.some((d) => shorteners.has(d)) ? 1 : 0;
      f.url_suspicious_tld = domains.some((d) => tlds.some((t) => d.endsWith(t))) ? 1 : 0;
      f.url_raw_ip = ipv4Re.test(text) ? 1 : 0;

      let lookalike = 0;
      for (const d of domains) {
        if (legit.has(d) || [...legit].some((g) => d.endsWith("." + g))) continue;
        const flat = d.replace(/[^a-z]/g, "");
        if (brandTokens.some((b) => flat.includes(b))) lookalike = 1;
      }
      f.url_brand_lookalike = lookalike;

      const phones = allMatches(phoneRe, norm);
      f.phone_present = phones.length ? 1 : 0;
      f.phone_count = phones.length;
      for (const p of phones) {
        if (offsets.length) {
          spans.push([offsets[p.start], offsets[Math.min(p.end - 1, offsets.length - 1)] + 1, "phone"]);
        }
      }

      const moneys = allMatches(moneyRe, norm);
      let big = 0;
      for (const m of moneys) {
        const digits = m.text.replace(/[^\d]/g, "");
        if (digits) big = Math.max(big, parseInt(digits, 10));
        if (offsets.length) {
          spans.push([offsets[m.start], offsets[Math.min(m.end - 1, offsets.length - 1)] + 1, "money"]);
        }
      }
      f.money_present = moneys.length ? 1 : 0;
      f.money_magnitude = big < 1000 ? 0 : Math.min(String(big).length - 3, 6);

      const lexOrder = [
        ["urgency", cfg.lexicons.urgency],
        ["prize", cfg.lexicons.prize],
        ["sensitive", cfg.lexicons.sensitive],
        ["asks_for_code", cfg.lexicons.asks_for_code],
        ["provides_code", cfg.lexicons.provides_code],
        ["action_link", cfg.lexicons.action_link],
        ["pay_fee", cfg.lexicons.pay_fee],
      ];
      for (const [name, lex] of lexOrder) {
        f["lex_" + name] = lexHits(norm, offsets, lex, name, spans);
      }
      f.otp_asks_not_provides =
        f.lex_asks_for_code > 0 && f.lex_provides_code === 0 ? 1 : 0;

      let brandHits = 0;
      for (const key of Object.keys(cfg.brands)) {
        const h = lexHits(norm, offsets, cfg.brands[key], "brand", spans);
        if (h) brandHits += 1;
      }
      f.brand_mention = brandHits;
      f.brand_and_link = brandHits > 0 && f.url_present > 0 ? 1 : 0;
      f.brand_and_lookalike = brandHits > 0 && lookalike > 0 ? 1 : 0;
      f.brand_and_urgency = brandHits > 0 && f.lex_urgency > 0 ? 1 : 0;

      const n = Math.max(text.length, 1);
      f.length = text.length;
      let digits = 0;
      for (const c of norm) if (c >= "0" && c <= "9") digits++;
      f.digit_ratio = digits / Math.max(norm.length, 1);
      f.exclaim_count = (text.match(/!/g) || []).length;
      let punct = 0;
      for (const c of text) if ("!?.,:*#".includes(c)) punct++;
      f.punct_density = punct / n;
      let upper = 0;
      for (const c of text) if (c !== c.toLowerCase() && c === c.toUpperCase()) upper++;
      f.upper_ratio = upper / n;

      const ar = (text.match(arabicRe) || []).length;
      const la = (text.match(latinRe) || []).length;
      const tot = ar + la;
      f.script_mix_ratio = tot ? Math.min(ar, la) / tot : 0;

      return { features: f, spans: spans };
    };
  }

  // ---- TF-IDF ---------------------------------------------------------------

  function tfidfVector(terms, index, idf, sublinear) {
    const counts = new Map();
    for (const t of terms) {
      const j = index.get(t);
      if (j === undefined) continue;
      counts.set(j, (counts.get(j) || 0) + 1);
    }
    let norm = 0;
    const vec = new Map();
    for (const [j, c] of counts) {
      const tf = sublinear ? 1 + Math.log(c) : c;
      const v = tf * idf[j];
      vec.set(j, v);
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (norm > 0) for (const [j, v] of vec) vec.set(j, v / norm);
    return vec;
  }

  // ---- engine ---------------------------------------------------------------

  function Engine(model) {
    this.model = model;
    this.normalizeTraced = buildNormalizer(model.normalize);
    this.analyse = buildSignals(model.signals, model.regex);
    this.urlRe = compile(model.regex.url, "gi");
    this.phoneRe = compile(model.regex.phone, "g");
    this.moneyRe = compile(model.regex.money, "gi");
    this.longNumRe = compile(model.regex.long_num, "g");

    this.index = {};
    for (const name of ["char", "word"]) {
      const b = model.blocks[name];
      const m = new Map();
      for (let i = 0; i < b.terms.length; i++) m.set(b.terms[i], i);
      this.index[name] = m;
    }
  }

  Engine.prototype.foldEntities = function (text) {
    let t = text.replace(this.urlRe, " __url__ ");
    t = t.replace(this.phoneRe, " __phone__ ");
    t = t.replace(this.moneyRe, " __money__ ");
    t = t.replace(this.longNumRe, " __num__ ");
    return t.replace(/\s+/g, " ").trim();
  };

  Engine.prototype.preprocess = function (text) {
    return this.foldEntities(this.normalizeTraced(text).text);
  };

  Engine.prototype.predict = function (text) {
    const m = this.model;
    const prepped = this.preprocess(text).toLowerCase();

    const contributions = [];
    let score = m.intercept;

    // char + word TF-IDF blocks
    for (const name of ["char", "word"]) {
      const b = m.blocks[name];
      const grams =
        b.analyzer === "char_wb"
          ? charWbNgrams(prepped, b.ngram_range[0], b.ngram_range[1])
          : wordNgrams(prepped, b.ngram_range[0], b.ngram_range[1]);
      const vec = tfidfVector(grams, this.index[name], b.idf, b.sublinear_tf);
      for (const [j, v] of vec) {
        const c = m.coef[b.offset + j] * v;
        score += c;
        if (Math.abs(c) > 1e-9) {
          contributions.push({ kind: name, name: b.terms[j], value: v, contribution: c });
        }
      }
    }

    // signals block
    const sb = m.blocks.signals;
    const res = this.analyse(text, this.normalizeTraced);
    for (let i = 0; i < sb.names.length; i++) {
      const raw = res.features[sb.names[i]] || 0;
      const v = sb.scale[i] ? raw / sb.scale[i] : 0;
      const c = m.coef[sb.offset + i] * v;
      score += c;
      if (Math.abs(c) > 1e-9) {
        contributions.push({
          kind: "signal", name: sb.names[i], value: raw, scaled: v, contribution: c,
        });
      }
    }

    const p = 1 / (1 + Math.exp(-score));
    contributions.sort((a, b2) => Math.abs(b2.contribution) - Math.abs(a.contribution));
    return {
      label: p >= 0.5 ? "scam" : "ham",
      probability: p,
      score: score,
      contributions: contributions,
      signals: res.features,
      spans: res.spans,
      normalized: this.normalizeTraced(text).text,
      folded: this.preprocess(text),
    };
  };

  global.ScamEngine = {
    Engine: Engine, charWbNgrams: charWbNgrams,
    wordNgrams: wordNgrams, pyRegexToJs: pyRegexToJs,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
