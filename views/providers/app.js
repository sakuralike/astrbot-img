const bridge = window.AstrBotPluginView;
const form = document.getElementById("provider-form");
const list = document.getElementById("provider-list");
const modelList = document.getElementById("model-list");
const feedback = document.getElementById("feedback");
const saveButton = document.getElementById("save-provider");
const fetchButton = document.getElementById("fetch-models");
const deleteButton = document.getElementById("delete-provider");

let providers = [];
let currentProviderId = "";

const fields = {
  originalId: document.getElementById("original-id"),
  id: document.getElementById("provider-id"),
  name: document.getElementById("name"),
  protocol: document.getElementById("protocol"),
  baseUrl: document.getElementById("base-url"),
  apiKey: document.getElementById("api-key"),
  model: document.getElementById("model"),
  modelsEndpoint: document.getElementById("models-endpoint"),
  enabled: document.getElementById("enabled"),
  allowHttp: document.getElementById("allow-http"),
  setCurrent: document.getElementById("set-current"),
  modelSearch: document.getElementById("model-search"),
};

function setFeedback(message, kind = "") {
  feedback.textContent = message;
  feedback.dataset.kind = kind;
}

function activeProvider(provider) {
  return provider.id === currentProviderId;
}

function renderProviders() {
  list.replaceChildren();
  if (!providers.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "还没有手动供应商";
    list.append(empty);
    return;
  }

  for (const provider of providers) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "provider-item";
    button.setAttribute("aria-current", String(provider.id === fields.id.value));
    const name = document.createElement("strong");
    name.textContent = provider.name || provider.id;
    const detail = document.createElement("small");
    detail.textContent = `${provider.protocol === "openai_responses" ? "Responses" : "Images"} · ${activeProvider(provider) ? "当前使用" : provider.enabled ? "已启用" : "已停用"}`;
    button.append(name, detail);
    button.addEventListener("click", () => loadProvider(provider));
    list.append(button);
  }
}

function renderModels(models = selectedProvider()?.models || []) {
  const query = fields.modelSearch.value.trim().toLowerCase();
  const filtered = models.filter((model) => model.toLowerCase().includes(query));
  modelList.replaceChildren();
  if (!filtered.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = models.length ? "没有匹配的模型" : "尚未获取模型，可在下方填写自定义模型 ID";
    modelList.append(empty);
    return;
  }
  for (const model of filtered) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-option";
    button.textContent = model;
    button.setAttribute("aria-pressed", String(fields.model.value === model));
    button.addEventListener("click", () => {
      fields.model.value = model;
      renderModels(models);
    });
    modelList.append(button);
  }
}

function selectedProvider() {
  return providers.find((provider) => provider.id === fields.originalId.value) || null;
}

function preferredProvider() {
  return providers.find((provider) => provider.id === currentProviderId)
    || providers.find((provider) => provider.enabled !== false)
    || providers[0]
    || null;
}

function loadProvider(provider) {
  fields.originalId.value = provider.id;
  fields.id.value = provider.id;
  fields.name.value = provider.name || "";
  fields.protocol.value = provider.protocol || "openai_images";
  fields.baseUrl.value = provider.base_url || "";
  fields.apiKey.value = "";
  fields.apiKey.placeholder = provider.has_api_key ? "已配置，留空则保留现有密钥" : "输入供应商 API Key";
  fields.model.value = provider.model || "";
  fields.modelsEndpoint.value = provider.models_endpoint || "";
  fields.enabled.checked = provider.enabled !== false;
  fields.allowHttp.checked = provider.allow_insecure_http === true;
  fields.setCurrent.checked = activeProvider(provider);
  document.getElementById("key-status").textContent = provider.has_api_key ? "密钥已配置" : "尚未配置";
  document.getElementById("form-heading").textContent = provider.name || "编辑供应商";
  deleteButton.hidden = false;
  setFeedback("");
  renderProviders();
  renderModels(provider.models || []);
}

function newProvider() {
  form.reset();
  fields.originalId.value = "";
  fields.id.value = "";
  fields.name.value = "";
  fields.protocol.value = "openai_images";
  fields.baseUrl.value = "";
  fields.apiKey.value = "";
  fields.apiKey.placeholder = "输入供应商 API Key";
  fields.model.value = "";
  fields.modelsEndpoint.value = "";
  fields.enabled.checked = true;
  fields.allowHttp.checked = false;
  fields.setCurrent.checked = true;
  document.getElementById("key-status").textContent = "尚未配置";
  document.getElementById("form-heading").textContent = "新增供应商";
  deleteButton.hidden = true;
  setFeedback("");
  renderProviders();
  renderModels([]);
}

function formPayload() {
  return {
    original_id: fields.originalId.value,
    id: fields.id.value.trim(),
    name: fields.name.value.trim(),
    protocol: fields.protocol.value,
    base_url: fields.baseUrl.value.trim(),
    api_key: fields.apiKey.value,
    model: fields.model.value.trim(),
    models: selectedProvider()?.models || [],
    models_endpoint: fields.modelsEndpoint.value.trim(),
    enabled: fields.enabled.checked,
    allow_insecure_http: fields.allowHttp.checked,
    set_current: fields.setCurrent.checked,
  };
}

async function saveProvider() {
  const result = await bridge.apiPost("providers/save", formPayload());
  if (result.provider) {
    const saved = result.provider;
    providers = providers.filter((provider) => provider.id !== fields.originalId.value && provider.id !== saved.id);
    providers.push(saved);
    currentProviderId = result.current_provider_id || currentProviderId;
    loadProvider(saved);
    fields.apiKey.value = "";
    fields.apiKey.placeholder = saved.has_api_key ? "已配置，留空则保留现有密钥" : "输入供应商 API Key";
    document.getElementById("key-status").textContent = saved.has_api_key ? "密钥已配置" : "尚未配置";
    renderProviders();
  }
  return result;
}

async function loadConfig() {
  await bridge.ready();
  const result = await bridge.apiGet("providers");
  providers = Array.isArray(result.providers) ? result.providers : [];
  currentProviderId = result.current_provider_id || "";
  document.getElementById("source-status").textContent = "仅使用插件手动供应商";
  renderProviders();
  const provider = preferredProvider();
  if (provider) loadProvider(provider);
  else newProvider();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  saveButton.disabled = true;
  setFeedback("正在保存…");
  try {
    await saveProvider();
    setFeedback("配置已保存", "success");
  } catch (error) {
    setFeedback(error.message || "保存失败", "error");
  } finally {
    saveButton.disabled = false;
  }
});

fetchButton.addEventListener("click", async () => {
  if (!form.reportValidity()) return;
  fetchButton.disabled = true;
  saveButton.disabled = true;
  setFeedback("正在保存并获取模型…");
  try {
    const saved = await saveProvider();
    const result = await bridge.apiPost("providers/models", { id: saved.provider.id });
    const updated = result.provider;
    providers = providers.map((provider) => provider.id === updated.id ? updated : provider);
    currentProviderId = saved.current_provider_id || currentProviderId;
    loadProvider(updated);
    setFeedback(`已获取 ${result.models.length} 个模型`, "success");
  } catch (error) {
    setFeedback(error.message || "获取模型失败，已保留原模型列表", "error");
  } finally {
    fetchButton.disabled = false;
    saveButton.disabled = false;
  }
});

deleteButton.addEventListener("click", async () => {
  const providerId = fields.originalId.value;
  if (!providerId || !window.confirm(`确定删除供应商“${fields.name.value}”吗？`)) return;
  deleteButton.disabled = true;
  setFeedback("正在删除…");
  try {
    const result = await bridge.apiPost("providers/delete", { id: providerId });
    providers = providers.filter((provider) => provider.id !== providerId);
    currentProviderId = result.current_provider_id || "";
    renderProviders();
    const provider = preferredProvider();
    if (provider) loadProvider(provider);
    else newProvider();
    setFeedback("供应商已删除", "success");
  } catch (error) {
    setFeedback(error.message || "删除失败", "error");
  } finally {
    deleteButton.disabled = false;
  }
});

document.getElementById("new-provider").addEventListener("click", newProvider);
fields.modelSearch.addEventListener("input", () => renderModels(selectedProvider()?.models || []));
fields.model.addEventListener("input", () => renderModels(selectedProvider()?.models || []));
fields.id.addEventListener("input", renderProviders);

loadConfig().catch((error) => {
  document.getElementById("source-status").textContent = "无法读取供应商配置";
  setFeedback(error.message || "页面初始化失败", "error");
});
