"use strict";

const $ = (selector) => document.querySelector(selector);
const BLANK = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
const KIND_TEXT = { attendance: "Chấm công", spoof: "Nghi giả mạo", unknown: "Người lạ" };
const LIVENESS_TEXT = { off: "Tắt", opportunistic: "Kiểm khi đủ điều kiện", required: "Bắt buộc" };
const livenessText = (value) => (value == null ? "chưa kiểm giả mạo" : `thật ${fixed(value)}`);

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));
const timeOf = (iso) => (iso ? iso.slice(11, 19) : "");
const fixed = (value) => (value == null ? "—" : Number(value).toFixed(2));

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const type = response.headers.get("content-type") || "";
  const body = type.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error((body && body.detail) || body || response.statusText);
  return body;
}

function toast(text, kind = "info") {
  const el = $("#toast");
  el.textContent = text;
  el.className = `toast ${kind}`;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 4000);
}

// ---------- tab ----------
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${name}`));
  // Chỉ giữ 1 luồng MJPEG khi đang xem tab Trực tiếp.
  $("#live-img").src = name === "live" ? "/stream.mjpg" : BLANK;
  if (name === "employees") loadEmployees().catch((e) => toast(e.message, "bad"));
  if (name === "history") loadHistory().catch((e) => toast(e.message, "bad"));
  if (name === "rejected") loadRejected().catch((e) => toast(e.message, "bad"));
}

// ---------- trạng thái ----------
let datesReady = false;
async function refreshStatus() {
  const badge = $("#cam-badge");
  try {
    const s = await api("/api/status");
    const cam = s.camera;
    badge.textContent = cam.connected ? `Camera: đang chạy (${cam.fps} fps)` : cam.ended ? "Video đã phát hết" : "Mất kết nối camera";
    badge.className = `badge ${cam.connected ? "ok" : "bad"}`;
    $("#status-list").innerHTML = [
      ["Nguồn", cam.source],
      ["Camera ID", s.camera_id],
      ["Độ phân giải", cam.width ? `${cam.width}×${cam.height}` : "—"],
      ["FPS camera / xử lý", `${cam.fps} / ${s.processing_fps}`],
      ["Model nhận diện", s.recognition_model === "arcface" ? "ArcFace (chỉ đánh giá)" : "SFace"],
      ["Chống giả mạo", LIVENESS_TEXT[s.liveness_mode] || s.liveness_mode],
      ["Nhân viên", s.employees],
      ["Lỗi gần nhất", cam.last_error || "—"],
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
    $("#recent-list").innerHTML = s.recent.length
      ? s.recent.map((r) => `<li class="${esc(r.kind)}"><b>${esc(r.kind === "attendance" ? r.name : KIND_TEXT[r.kind])}</b>
          <span>${esc(KIND_TEXT[r.kind])} · ${timeOf(r.ts)} · khớp ${fixed(r.score)}</span></li>`).join("")
      : `<li class="empty">Chưa có</li>`;
    if (!datesReady) {
      datesReady = true;
      const day = s.now.slice(0, 10);
      ["#hist-date", "#csv-start", "#csv-end", "#rej-date"].forEach((id) => { $(id).value = day; });
    }
  } catch (e) {
    badge.textContent = "Mất kết nối máy chủ";
    badge.className = "badge bad";
  }
}

// ---------- nhân viên ----------
async function loadEmployees() {
  const list = await api("/api/employees");
  $("#emp-body").innerHTML = list.length
    ? list.map((e) => `<tr><td>${esc(e.code)}</td><td>${esc(e.name)}</td>
        <td><span class="pill ${e.sample_count >= 3 ? "ok" : "warn"}">${e.sample_count}</span></td>
        <td class="actions">
          <button data-act="enroll" data-id="${e.id}" data-name="${esc(e.name)}">Đăng ký khuôn mặt</button>
          <button data-act="delete" data-id="${e.id}" data-name="${esc(e.name)}" class="danger">Xoá</button>
        </td></tr>`).join("")
    : `<tr><td colspan="4" class="empty">Chưa có nhân viên</td></tr>`;
}

$("#add-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const data = new FormData(form);
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    await api("/api/employees", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: data.get("code"), name: data.get("name") }),
    });
    form.reset();
    toast("Đã thêm nhân viên", "ok");
    await loadEmployees();
  } catch (e) {
    toast(e.message, "bad");
  } finally {
    button.disabled = false;
  }
});

$("#emp-body").addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  const { id, name, act } = button.dataset;
  if (act === "enroll") return openEnroll(id, name);
  if (!confirm(`Xoá ${name}? Lịch sử chấm công được giữ lại; ảnh và đặc trưng khuôn mặt bị xoá vĩnh viễn.`)) return;
  try {
    await api(`/api/employees/${id}`, { method: "DELETE" });
    toast("Đã xoá nhân viên", "ok");
    await loadEmployees();
  } catch (e) {
    toast(e.message, "bad");
  }
});

// ---------- đăng ký khuôn mặt ----------
let enrollId = null;

function openEnroll(id, name) {
  enrollId = id;
  $("#enroll-title").textContent = `Đăng ký khuôn mặt — ${name}`;
  $("#enroll-msg").textContent = "Đứng/ngồi để thấy trọn khuôn mặt (không bị cắt mép), đủ sáng. Bấm chụp rồi "
    + "XOAY ĐẦU CHẬM: trái, phải, ngẩng, cúi — chỉ mẫu khác góc mới được lưu. Nên chụp bằng chính camera lắp thật.";
  $("#enroll-img").src = "/stream.mjpg";
  $("#enroll-dialog").showModal();
  loadSamples().catch((e) => toast(e.message, "bad"));
}

$("#enroll-dialog").addEventListener("close", () => {
  $("#enroll-img").src = BLANK;
  enrollId = null;
  loadEmployees().catch(() => {});
});

async function loadSamples() {
  const samples = await api(`/api/employees/${enrollId}/samples`);
  $("#samples-grid").innerHTML = samples.length
    ? samples.map((s) => `<figure><img src="/files/${esc(s.image_path)}" alt="">
        <figcaption>${s.source === "camera" ? "Camera" : "Tải lên"}
        <button data-sample="${s.id}" class="icon danger" title="Xoá mẫu">✕</button></figcaption></figure>`).join("")
    : `<p class="empty">Chưa có mẫu</p>`;
}

$("#capture-btn").addEventListener("click", async () => {
  const button = $("#capture-btn");
  button.disabled = true;
  $("#enroll-msg").textContent = "Đang chụp… giữ yên, nhìn thẳng camera.";
  try {
    const result = await api(`/api/employees/${enrollId}/capture`, { method: "POST" });
    const skipped = result.reasons.map((r) => `${r.text} (${r.count})`).join("; ");
    $("#enroll-msg").textContent = `Đã lưu ${result.saved} mẫu.` + (skipped ? ` Bỏ qua: ${skipped}` : "");
    await loadSamples();
  } catch (e) {
    $("#enroll-msg").textContent = e.message;
  } finally {
    button.disabled = false;
  }
});

$("#upload-input").addEventListener("change", async (event) => {
  const files = [...event.target.files];
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  $("#enroll-msg").textContent = "Đang xử lý ảnh…";
  try {
    const results = await api(`/api/employees/${enrollId}/upload`, { method: "POST", body: form });
    $("#enroll-msg").innerHTML = results
      .map((r) => `<div class="${r.ok ? "ok" : "bad"}">${esc(r.filename)}: ${esc(r.message)}</div>`).join("");
    await loadSamples();
  } catch (e) {
    $("#enroll-msg").textContent = e.message;
  } finally {
    event.target.value = "";
  }
});

$("#samples-grid").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-sample]");
  if (!button || !confirm("Xoá mẫu này?")) return;
  try {
    await api(`/api/samples/${button.dataset.sample}`, { method: "DELETE" });
    await loadSamples();
  } catch (e) {
    toast(e.message, "bad");
  }
});

// ---------- lịch sử ----------
function card(path, title, meta) {
  const image = path ? `<img loading="lazy" src="/files/${esc(path)}" alt="">` : `<div class="noimg">Không còn ảnh</div>`;
  return `<figure class="card">${image}<figcaption><b>${title}</b><span>${meta}</span></figcaption></figure>`;
}

async function loadHistory() {
  const day = $("#hist-date").value;
  if (!day) return;
  const r = await api(`/api/attendance?date=${day}`);
  $("#summary-body").innerHTML = r.summary.length
    ? r.summary.map((s) => `<tr><td>${esc(s.code)}</td><td>${esc(s.name)}</td><td>${timeOf(s.first_ts)}</td>
        <td>${s.count > 1 ? timeOf(s.last_ts) : "—"}</td><td>${s.count}</td></tr>`).join("")
    : `<tr><td colspan="5" class="empty">Không có lần chấm nào</td></tr>`;
  $("#events-grid").innerHTML = r.events.slice().reverse().map((e) => card(
    e.snapshot_path,
    `${esc(e.name)} <span class="muted">${esc(e.code)}</span>`,
    `${timeOf(e.ts)} · khớp ${fixed(e.score)} · ${livenessText(e.liveness_score)}`,
  )).join("") || `<p class="empty">—</p>`;
}

async function loadRejected() {
  const day = $("#rej-date").value;
  if (!day) return;
  const r = await api(`/api/rejected?date=${day}`);
  $("#rejected-grid").innerHTML = r.events.map((e) => card(
    e.snapshot_path,
    esc(KIND_TEXT[e.reason] || e.reason),
    `${timeOf(e.ts)}${e.best_name ? ` · gần nhất: ${esc(e.best_name)} (${fixed(e.score)})` : ""} · ${livenessText(e.liveness_score)}`,
  )).join("") || `<p class="empty">Không có</p>`;
}

$("#hist-date").addEventListener("change", () => loadHistory().catch((e) => toast(e.message, "bad")));
$("#hist-refresh").addEventListener("click", () => loadHistory().catch((e) => toast(e.message, "bad")));
$("#rej-date").addEventListener("change", () => loadRejected().catch((e) => toast(e.message, "bad")));
$("#rej-refresh").addEventListener("click", () => loadRejected().catch((e) => toast(e.message, "bad")));
$("#csv-btn").addEventListener("click", () => {
  location.href = `/api/attendance.csv?start=${$("#csv-start").value}&end=${$("#csv-end").value}`;
});

// ---------- khởi động ----------
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
refreshStatus().then(() => showTab("live"));
setInterval(refreshStatus, 1500);
