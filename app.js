const MESES = ["Set", "Ago", "Jul", "Jun", "Mai", "Abr"];
const PAYOPT = [[-2, "Sem consumo"], [-1, "Pago em dia"], [0, "Uso rotativo"], [1, "1 mês de atraso"], [2, "2 meses"], [3, "3 meses"], [4, "4 meses"], [5, "5 meses"], [6, "6 meses"], [7, "7 meses"], [8, "8+ meses"]];
const $ = id => document.getElementById(id);
const history = [];
const modelInfo = new Map();
let lastProbability;
let requestController;
let revision = 0;
let referencePreviouslyUsed = false;

$("abrir-ajuda").onclick = () => $("ajuda").showModal();
const navLinks = document.querySelectorAll(".nav a");
function updateNavigation() {
  const activeHash = [...navLinks].some(link => link.hash === location.hash) ? location.hash : "#painel";
  for (const link of navLinks) {
    if (link.hash === activeHash) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
}
window.addEventListener("hashchange", updateNavigation);
updateNavigation();

const grid = $("months");
grid.insertAdjacentHTML("beforeend", '<div></div>' + MESES.map(m => `<div class="h">${m}/2005</div>`).join(""));
grid.insertAdjacentHTML("beforeend", '<div class="r">Situação</div>' + MESES.map((m, i) => `<select id="pay${i}" aria-label="Situação em ${m}">${PAYOPT.map(([v, t]) => `<option value="${v}"${v === 0 ? " selected" : ""}>${t}</option>`).join("")}</select>`).join(""));
grid.insertAdjacentHTML("beforeend", '<div class="r">Fatura</div>' + MESES.map((m, i) => `<input id="bill${i}" type="number" step="0.01" required value="20000" aria-label="Fatura em ${m}">`).join(""));
grid.insertAdjacentHTML("beforeend", '<div class="r">Pago</div>' + MESES.map((m, i) => `<input id="paid${i}" type="number" min="0" step="0.01" required value="2000" aria-label="Pago em ${m}">`).join(""));

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function invalidate() {
  revision += 1;
  requestController?.abort();
  lastProbability = undefined;
  $("pct").textContent = "–";
  $("arc").setAttribute("stroke-dasharray", "0 100");
  $("verdict").textContent = "Aguardando análise";
  $("sub").textContent = "Dados alterados. Clique em analisar.";
  $("fx").replaceChildren();
}

function read() {
  const number = id => Number($(id).value);
  const row = prefix => MESES.map((_, i) => number(prefix + i));
  return {limit: number("limit"), age: number("age"), sex: number("sex"), edu: number("edu"), mar: number("mar"),
    pay: row("pay"), bill: row("bill"), paid: row("paid"), model: $("model").value, threshold: number("thr") / 100};
}

async function predict(input, signal) {
  const response = await fetch("/predict", {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(input), signal,
  });
  if (!response.ok) {
    if (response.status === 422) throw new Error("Confira os valores informados e os seis meses do histórico.");
    if (response.status === 503) throw new Error("Modelo indisponível. Execute o treinamento e reinicie o servidor.");
    throw new Error("A análise falhou. Consulte o servidor e tente novamente.");
  }
  const data = await response.json();
  if (!Number.isFinite(data.prob) || data.prob < 0 || data.prob > 1 || !Array.isArray(data.factors)) {
    throw new Error("O servidor retornou uma resposta inválida.");
  }
  return data;
}

function chart() {
  const input = read();
  const bills = [...input.bill].reverse(), payments = [...input.paid].reverse();
  const min = Math.min(0, ...bills, ...payments);
  const max = Math.max(1, ...bills, ...payments);
  const y = value => 150 - (value - min) / (max - min) * 140;
  const zero = y(0);
  let svg = [0, 0.5, 1].map(t => `<line x1="40" x2="550" y1="${150 - t * 140}" y2="${150 - t * 140}" stroke="#ececee"/><text x="34" y="${154 - t * 140}" text-anchor="end">${Math.round((min + (max - min) * t) / 1000)}k</text>`).join("");
  [...MESES].reverse().forEach((month, i) => {
    const x = 70 + i * 84;
    svg += `<rect x="${x - 22}" y="${Math.min(y(bills[i]), zero)}" width="20" height="${Math.abs(zero - y(bills[i]))}" rx="3" fill="#111"/>`
      + `<rect x="${x + 2}" y="${Math.min(y(payments[i]), zero)}" width="20" height="${Math.abs(zero - y(payments[i]))}" rx="3" fill="#c9c9cd"/>`
      + `<text x="${x}" y="170" text-anchor="middle">${month}</text>`;
  });
  $("chart").innerHTML = svg;
}

function renderClassification(probability) {
  const high = probability >= Number($("thr").value) / 100;
  $("verdict").textContent = high ? "Provável inadimplente" : "Provável adimplente";
  $("sub").textContent = high ? "A probabilidade atingiu o limiar escolhido." : "A probabilidade está abaixo do limiar escolhido.";
  return high;
}

function renderFactors(factors) {
  $("fx").replaceChildren();
  for (const factor of [...factors].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 3)) {
    const item = document.createElement("li");
    const name = document.createElement("span");
    const value = document.createElement("b");
    name.textContent = factor.name;
    value.textContent = `${factor.value > 0 ? "+" : ""}${(factor.value * 100).toLocaleString("pt-BR", {maximumFractionDigits: 1})} p.p.`;
    item.append(name, value);
    $("fx").append(item);
  }
}

function renderHistory() {
  $("hist").replaceChildren();
  for (const entry of history) {
    const item = document.createElement("li"), detail = document.createElement("span");
    detail.textContent = `Limite NT$ ${entry.limit.toLocaleString("pt-BR")}`;
    const note = document.createElement("small");
    note.textContent = `${entry.time} · ${entry.percent}% · ${entry.model} · limiar ${entry.threshold}%`;
    detail.append(note);
    const tag = document.createElement("span");
    tag.className = `tag${entry.high ? " in" : ""}`;
    tag.textContent = entry.high ? "Inadimplente" : "Adimplente";
    item.append(detail, tag);
    $("hist").append(item);
  }
}

async function analyze() {
  for (const field of document.querySelectorAll("#clientes input, #clientes select")) {
    if (!field.reportValidity()) return;
  }
  invalidate();
  const current = revision;
  const input = read();
  const controller = new AbortController();
  requestController = controller;
  const timeout = setTimeout(() => controller.abort(), 15000);
  $("run").disabled = true;
  $("run").textContent = "Analisando…";
  status("Calculando previsão…");
  try {
    const result = await predict(input, controller.signal);
    if (revision !== current) return;
    lastProbability = result.prob;
    const percent = (result.prob * 100).toLocaleString("pt-BR", {maximumFractionDigits: 1});
    $("pct").textContent = `${percent}%`;
    $("arc").setAttribute("stroke-dasharray", `${result.prob * 100} 100`);
    const high = renderClassification(result.prob);
    renderFactors(result.factors);
    history.unshift({time: new Date().toLocaleTimeString("pt-BR", {hour: "2-digit", minute: "2-digit"}),
      percent, high, limit: input.limit, model: result.model_used, threshold: input.threshold * 100});
    history.length = Math.min(history.length, 5);
    renderHistory();
    status(`Análise concluída com ${result.model_used}. Base UCI: Taiwan, 2005; valores em NT$.`);
    if (window.matchMedia("(max-width: 980px)").matches) location.hash = "#analises";
  } catch (error) {
    if (revision !== current) return;
    status(error.name === "AbortError" ? "O servidor demorou a responder. Tente novamente." : (error.message || "Não foi possível conectar ao servidor."), true);
    $("verdict").textContent = "Análise indisponível";
    $("sub").textContent = "Não foi possível obter uma previsão.";
  } finally {
    clearTimeout(timeout);
    $("run").disabled = modelInfo.size === 0;
    $("run").textContent = "Analisar cliente";
    if (requestController === controller) requestController = undefined;
  }
}

function selectModel() {
  invalidate();
  const model = modelInfo.get($("model").value);
  if (!model) return;
  $("thr").value = (model.threshold * 100).toFixed(1);
  $("thv").textContent = `${Number($("thr").value).toLocaleString("pt-BR")}%`;
  const metrics = model.test_metrics;
  const calibrationNames = {none: "sem ajuste adicional", sigmoid: "sigmoide", isotonic: "isotônica"};
  const evaluation = referencePreviouslyUsed ? "Referência já consultada" : "Teste";
  $("model-metrics").textContent = `${evaluation}: AUC-ROC ${metrics.roc_auc.toFixed(3)} · AP ${metrics.average_precision.toFixed(3)} · F1 ${metrics.f1.toFixed(3)} (limiar validado: ${(model.threshold * 100).toFixed(1)}%). Calibração: ${calibrationNames[model.calibration_method] || "sem ajuste adicional"}.`;
  status(`Modelo: ${model.name}. Pronto para analisar. Estudo acadêmico com dados de Taiwan, 2005.`);
}

const PRESETS = {
  good: {limit: 200000, age: 42, pay: [-1, -1, -1, -1, -1, -1], bill: [12000, 11000, 9000, 10000, 8000, 9500], paid: [12000, 11000, 9000, 10000, 8000, 9500]},
  bad: {limit: 30000, age: 27, pay: [3, 3, 2, 2, 2, 0], bill: [29500, 28000, 27000, 26000, 25000, 22000], paid: [0, 0, 1000, 0, 500, 1500]},
};
document.querySelectorAll("[data-preset]").forEach(button => button.onclick = async () => {
  if (!modelInfo.size) return;
  invalidate();
  const preset = PRESETS[button.dataset.preset];
  $("limit").value = preset.limit;
  $("age").value = preset.age;
  MESES.forEach((_, i) => {
    $("pay" + i).value = preset.pay[i];
    $("bill" + i).value = preset.bill[i];
    $("paid" + i).value = preset.paid[i];
  });
  chart();
  await analyze();
});
document.querySelectorAll("#clientes input, #clientes select").forEach(field => field.addEventListener("input", () => {
  invalidate();
  chart();
  if (modelInfo.size) status("Dados alterados. Execute uma nova análise.");
}));
$("run").onclick = analyze;
$("model").onchange = selectModel;
$("thr").oninput = event => {
  $("thv").textContent = `${Number(event.target.value).toLocaleString("pt-BR")}%`;
  if (lastProbability !== undefined) renderClassification(lastProbability);
  else if (requestController) {
    invalidate();
    status("Limiar alterado. Execute uma nova análise.");
  }
};

async function initialize() {
  try {
    const response = await fetch("/models", {signal: AbortSignal.timeout(15000)});
    if (!response.ok) throw new Error("Modelos indisponíveis. Execute python train.py, reinicie o servidor e recarregue a página.");
    const data = await response.json();
    referencePreviouslyUsed = Boolean(data.reference_previously_used);
    $("model").replaceChildren();
    for (const model of data.models) {
      modelInfo.set(model.id, model);
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.name + (model.id === data.selected_model ? " (selecionado na validação)" : "");
      $("model").append(option);
    }
    $("model").value = data.selected_model;
    $("model").disabled = false;
    $("run").disabled = false;
    selectModel();
  } catch (error) {
    status(error.message || "Não foi possível conectar ao servidor. Abra o aplicativo por http://127.0.0.1:8000.", true);
  }
}
chart();
initialize();
