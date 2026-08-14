/**
 * AGLM Universal Tokenizer Inspector — Live 3-Way Split Screen Controller
 */

// Presets data
const SAMPLE_PRESETS = {
  hi: "नमस्ते, मेरा नाम आकाश है। हम एक नया बहुभाषी टोकनाइज़र बना रहे हैं।",
  hinglish: "mujhe ye kaam complete karna hai, sab theek chal raha hai aur hum aage badh rahe hain.",
  te: "నమస్కారం, నా పేరు ఆకాష్. మేము సరికొత్త భాషా మోഡల్ శిక్షణ ఇస్తున్నాము.",
  tenglish: "nenu oka kottha multilingual language model train chestunnanu. ee model telugu tho paatu english technical words ni kuda sarigga understand chesukovali.",
  ta: "வணக்கம், என் பெயர் ஆகாஷ். நாங்கள் புதிய பலமொழி டோக்கனைசர் உருவாக்குகிறோம்.",
  tanglish: "naan oru pudhiya multilingual language model train pannitu irukken. indha model tamil mattum illama english technical words um nalla understand pannanum.",
  kn: "ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು ಆಕಾಶ್. ನಾವು ಹೊಸ ಬಹುಭಾಷಾ ಮಾದರಿಯನ್ನು ತರಬೇತಿ ಮಾಡುತ್ತಿದ್ದೇವೆ.",
  kanglish: "naanu ondu hosa multilingual language model train maduttiddene. ee model kannada jothege english technical words mattu code mixed sentences annu sariyagi understand madabeku.",
  ml: "നമസ്കാരം, എൻ്റെ പേര് ആകാശ്. ഞങ്ങൾ ഒരു പുതിയ ഭാഷാ മോഡൽ പരിശീലിപ്പിക്കുന്നു.",
  manglish: "njan oru puthiya multilingual language model train cheyyukayaanu. ee model malayalam mathramalla english technical terms um code mixed sentences um nannayi understand cheyyanam.",
  code: "def calculate_hash(data: bytes, salt: str = 'aglm_1m') -> str:\n    return hashlib.sha256(data + salt.encode()).hexdigest()",
  en: "The universal tokenizer efficiently represents agglutinative suffixes, code tokens, and conversational variations."
};

let debounceTimer = null;
let currentResults = {};
let activeSplitMode = "3way"; // '3way' or '4way'

// DOM Elements
const mainTextInput = document.getElementById("mainTextInput");
const clearInputBtn = document.getElementById("clearInputBtn");
const charCountBadge = document.getElementById("charCountBadge");
const byteCountBadge = document.getElementById("byteCountBadge");

// Scoreboard
const scoreboardWinnerTitle = document.getElementById("scoreboardWinnerTitle");
const scoreboardSavingsSummary = document.getElementById("scoreboardSavingsSummary");
const pillAglmVal = document.getElementById("pillAglmVal");
const pillOpenaiVal = document.getElementById("pillOpenaiVal");
const pillGemmaVal = document.getElementById("pillGemmaVal");

// Columns Elements
const aglmTokenCount = document.getElementById("aglmTokenCount");
const aglmBpt = document.getElementById("aglmBpt");
const aglmChipsBox = document.getElementById("aglmChipsBox");
const aglmSavingsBanner = document.getElementById("aglmSavingsBanner");

const openaiTokenCount = document.getElementById("openaiTokenCount");
const openaiBpt = document.getElementById("openaiBpt");
const openaiChipsBox = document.getElementById("openaiChipsBox");
const openaiDiffBanner = document.getElementById("openaiDiffBanner");

const gemmaTokenCount = document.getElementById("gemmaTokenCount");
const gemmaBpt = document.getElementById("gemmaBpt");
const gemmaChipsBox = document.getElementById("gemmaChipsBox");
const gemmaDiffBanner = document.getElementById("gemmaDiffBanner");

const aglm256Col = document.getElementById("aglm256Col");
const aglm256TokenCount = document.getElementById("aglm256TokenCount");
const aglm256Bpt = document.getElementById("aglm256Bpt");
const aglm256ChipsBox = document.getElementById("aglm256ChipsBox");
const aglm256DiffBanner = document.getElementById("aglm256DiffBanner");

const splitGridContainer = document.getElementById("splitGridContainer");
const modeSplit3Btn = document.getElementById("modeSplit3Btn");
const modeSplit4Btn = document.getElementById("modeSplit4Btn");

// Tooltip
const tooltip = document.getElementById("tokenInspectorTooltip");
const tooltipTokenId = document.getElementById("tooltipTokenId");
const tooltipScript = document.getElementById("tooltipScript");
const tooltipText = document.getElementById("tooltipText");
const tooltipHex = document.getElementById("tooltipHex");
const tooltipByteLen = document.getElementById("tooltipByteLen");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();

  // Set default initial Hindi prompt
  mainTextInput.value = SAMPLE_PRESETS.hi;
  triggerTokenize();
});

function setupEventListeners() {
  // Live input typing with debounce
  mainTextInput.addEventListener("input", () => {
    updateInputStats();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(triggerTokenize, 120);
  });

  // Clear button
  clearInputBtn.addEventListener("click", () => {
    mainTextInput.value = "";
    updateInputStats();
    triggerTokenize();
  });

  // Preset buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const lang = btn.dataset.lang;
      if (SAMPLE_PRESETS[lang]) {
        mainTextInput.value = SAMPLE_PRESETS[lang];
        updateInputStats();
        triggerTokenize();
      }
    });
  });

  // Split mode tabs
  modeSplit3Btn.addEventListener("click", () => {
    activeSplitMode = "3way";
    modeSplit3Btn.classList.add("active");
    modeSplit4Btn.classList.remove("active");
    splitGridContainer.className = "split-grid split-3col";
    aglm256Col.style.display = "none";
  });

  modeSplit4Btn.addEventListener("click", () => {
    activeSplitMode = "4way";
    modeSplit4Btn.classList.add("active");
    modeSplit3Btn.classList.remove("active");
    splitGridContainer.className = "split-grid split-4col";
    aglm256Col.style.display = "flex";
  });

  // Copy IDs buttons
  document.querySelectorAll(".copy-col-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const targetModel = btn.dataset.target;
      const res = currentResults[targetModel];
      if (res && res.tokens && res.tokens.length > 0) {
        const ids = res.tokens.map(t => t.id);
        navigator.clipboard.writeText(JSON.stringify(ids));
        const originalText = btn.textContent;
        btn.textContent = "✅ Copied!";
        setTimeout(() => btn.textContent = originalText, 1800);
      }
    });
  });
}

function updateInputStats() {
  const text = mainTextInput.value;
  const chars = text.length;
  const bytes = new TextEncoder().encode(text).length;
  charCountBadge.textContent = `${chars.toLocaleString()} characters`;
  byteCountBadge.textContent = `${bytes.toLocaleString()} bytes (UTF-8)`;
}

async function triggerTokenize() {
  const text = mainTextInput.value;
  updateInputStats();

  if (!text.trim()) {
    renderEmptyState();
    return;
  }

  const requestedModels = ["aglm_1m", "o200k_base", "gemma2", "aglm_256k"];

  try {
    const res = await fetch("/api/tokenize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        model: "aglm_1m",
        compare_models: requestedModels
      })
    });

    const data = await res.json();
    currentResults = data.results || {};
    renderSplitResults(currentResults);
  } catch (err) {
    console.error("Tokenization error:", err);
  }
}

function renderSplitResults(results) {
  const aglmRes = results["aglm_1m"];
  const openaiRes = results["o200k_base"];
  const gemmaRes = results["gemma2"];
  const aglm256Res = results["aglm_256k"];

  if (!aglmRes) return;

  const aCount = aglmRes.token_count || 0;
  const oCount = openaiRes ? (openaiRes.token_count || 0) : 0;
  const gCount = gemmaRes ? (gemmaRes.token_count || 0) : 0;
  const a256Count = aglm256Res ? (aglm256Res.token_count || 0) : 0;

  // 1. Update Scoreboard & Savings
  pillAglmVal.textContent = `${aCount} toks`;
  pillOpenaiVal.textContent = oCount ? `${oCount} toks` : "--";
  pillGemmaVal.textContent = gCount ? `${gCount} toks` : "--";

  let savingsSummary = [];
  if (oCount > aCount) {
    const pctO = Math.round(((oCount - aCount) / oCount) * 100);
    savingsSummary.push(`🔥 <b>${pctO}% fewer tokens</b> than OpenAI GPT-4o`);
  }
  if (gCount > aCount) {
    const pctG = Math.round(((gCount - aCount) / gCount) * 100);
    savingsSummary.push(`💎 <b>${pctG}% fewer tokens</b> than Google Gemma 2`);
  }

  if (savingsSummary.length > 0) {
    scoreboardWinnerTitle.textContent = "🏆 AGLM Universal (Ours) Wins with Highest Compression Efficiency!";
    scoreboardSavingsSummary.innerHTML = savingsSummary.join(" • ");
    aglmSavingsBanner.innerHTML = `👑 Optimal Density: Saves ${savingsSummary.length > 1 ? 'up to ' + Math.max(Math.round(((oCount - aCount) / oCount) * 100), Math.round(((gCount - aCount) / gCount) * 100)) + '%' : 'tokens'} vs baseline`;
  } else {
    scoreboardWinnerTitle.textContent = "⚡ Real-Time Multi-Model Token Density Comparison";
    scoreboardSavingsSummary.textContent = "Comparing AGLM 1.55M vs OpenAI GPT-4o and Google Gemma 2.";
    aglmSavingsBanner.textContent = "⚡ Best-in-class multi-tokenizer representation";
  }

  // 2. Render Column 1: AGLM 1.55M Max
  aglmTokenCount.textContent = aCount;
  aglmBpt.textContent = (aglmRes.bytes_per_token || 0.0).toFixed(2);
  renderChipsIntoBox(aglmChipsBox, aglmRes.tokens);

  // 3. Render Column 2: OpenAI GPT-4o
  if (openaiRes && !openaiRes.error) {
    openaiTokenCount.textContent = oCount;
    openaiBpt.textContent = (openaiRes.bytes_per_token || 0.0).toFixed(2);
    renderChipsIntoBox(openaiChipsBox, openaiRes.tokens);
    if (oCount > aCount) {
      const diff = oCount - aCount;
      openaiDiffBanner.textContent = `+${diff} more tokens than AGLM (+${Math.round((diff / aCount) * 100)}% token overhead)`;
    } else {
      openaiDiffBanner.textContent = "OpenAI Production Flagship Tokenizer";
    }
  }

  // 4. Render Column 3: Google Gemma 2
  if (gemmaRes && !gemmaRes.error) {
    gemmaTokenCount.textContent = gCount;
    gemmaBpt.textContent = (gemmaRes.bytes_per_token || 0.0).toFixed(2);
    renderChipsIntoBox(gemmaChipsBox, gemmaRes.tokens);
    if (gCount > aCount) {
      const diff = gCount - aCount;
      gemmaDiffBanner.textContent = `+${diff} more tokens than AGLM (+${Math.round((diff / aCount) * 100)}% token overhead)`;
    } else {
      gemmaDiffBanner.textContent = "Google Gemma 2 9B Tokenizer";
    }
  }

  // 5. Render Column 4: AGLM 256K
  if (aglm256Res && !aglm256Res.error) {
    aglm256TokenCount.textContent = a256Count;
    aglm256Bpt.textContent = (aglm256Res.bytes_per_token || 0.0).toFixed(2);
    renderChipsIntoBox(aglm256ChipsBox, aglm256Res.tokens);
    aglm256DiffBanner.textContent = "Compact 256K Balanced Production Tier";
  }
}

function renderChipsIntoBox(container, tokensList) {
  container.innerHTML = "";
  if (!tokensList || tokensList.length === 0) {
    container.innerHTML = `<div class="empty-state">No tokens</div>`;
    return;
  }

  tokensList.forEach(tok => {
    const chip = document.createElement("span");
    chip.className = "token-chip";
    chip.style.backgroundColor = tok.color.bg;
    chip.style.color = tok.color.text;
    chip.style.border = `1px solid ${tok.color.border}`;

    // Handle leading whitespace visibility
    if (tok.text.startsWith(" ")) {
      const spaceDot = document.createElement("span");
      spaceDot.className = "token-chip-space";
      spaceDot.textContent = "␣";
      chip.appendChild(spaceDot);
      chip.appendChild(document.createTextNode(tok.text.slice(1)));
    } else {
      chip.textContent = tok.text;
    }

    // Tooltip listeners
    chip.addEventListener("mouseenter", (e) => showTooltip(e, tok));
    chip.addEventListener("mousemove", (e) => moveTooltip(e));
    chip.addEventListener("mouseleave", hideTooltip);

    container.appendChild(chip);
  });
}

function renderEmptyState() {
  pillAglmVal.textContent = "--";
  pillOpenaiVal.textContent = "--";
  pillGemmaVal.textContent = "--";

  aglmTokenCount.textContent = "0";
  aglmBpt.textContent = "0.00";
  aglmChipsBox.innerHTML = `<div class="empty-state">Type above to tokenize</div>`;

  openaiTokenCount.textContent = "0";
  openaiBpt.textContent = "0.00";
  openaiChipsBox.innerHTML = `<div class="empty-state">Type above to tokenize</div>`;

  gemmaTokenCount.textContent = "0";
  gemmaBpt.textContent = "0.00";
  gemmaChipsBox.innerHTML = `<div class="empty-state">Type above to tokenize</div>`;

  aglm256TokenCount.textContent = "0";
  aglm256Bpt.textContent = "0.00";
  aglm256ChipsBox.innerHTML = `<div class="empty-state">Type above to tokenize</div>`;
}

// Tooltip helpers
function showTooltip(e, tok) {
  tooltipTokenId.textContent = `#${tok.id}`;
  tooltipScript.textContent = tok.script || "Latin";
  tooltipText.textContent = JSON.stringify(tok.text);
  tooltipHex.textContent = tok.bytes_hex ? `0x${tok.bytes_hex}` : "N/A";
  tooltipByteLen.textContent = `${tok.byte_len} bytes`;

  tooltip.style.display = "block";
  moveTooltip(e);
}

function moveTooltip(e) {
  const x = e.clientX + 14;
  const y = e.clientY + 14;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

function hideTooltip() {
  tooltip.style.display = "none";
}
