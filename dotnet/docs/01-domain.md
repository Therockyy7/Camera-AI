# Bài 01 — Tầng Domain: lõi nghiệp vụ thuần

> Project: `src/CameraAI.Domain` · Test: `tests/CameraAI.Domain.Tests` (36 test)
>
> Chạy test: `cd D:\CameraAI\dotnet` rồi `dotnet test`

---

## 1. Domain chứa gì — và KHÔNG chứa gì

**Chứa:** quy tắc nghiệp vụ trả lời câu hỏi *"khi nào thì được ghi chấm công?"*.
**Không chứa:** OpenCV, ONNX, SQLite, HTTP, `appsettings.json`, `ILogger`. Mở `CameraAI.Domain.csproj` — không có
một `PackageReference` nào. Đó là dấu hiệu Domain sạch.

Hệ quả thực tế: 36 test chạy trong ~200 ms, **không cần camera, không cần model, không cần DB**. Đây là điểm
"Untestability" trong phần *Software Rot* của file ôn tập — ta đã tránh được nó ngay từ thiết kế.

```text
src/CameraAI.Domain/
├── Common/DomainException.cs        lỗi nghiệp vụ (Web dịch thành 400 / 409)
├── Employees/
│   ├── Employee.cs                  Entity — rich model
│   └── FaceSample.cs                Entity + SampleEmbedding (value object)
├── Faces/
│   ├── FaceBox.cs                   Value object (record struct) + IoU
│   ├── FaceObservation.cs           kết quả 1 mặt / 1 frame + FaceMatch + RejectReason
│   ├── QualityGates.cs              RecognitionGate, LivenessGate
│   └── FaceGallery.cs               so khớp cosine, thread-safe không lock
└── Attendance/
    ├── AttendanceEvent.cs           AttendanceEvent, RejectedEvent
    ├── FaceTracker.cs               Track + FaceTracker
    ├── LivenessPolicies.cs          ★ Strategy + Factory Method
    └── AttendanceDecider.cs         ★ trái tim: bỏ phiếu → quyết định
```

**Thứ tự đọc gợi ý:** `FaceBox` → `FaceObservation` → `FaceTracker` → `LivenessPolicies` → `AttendanceDecider`
→ mở `AttendanceTests.cs` đọc song song.

---

## 2. Luồng một frame đi qua Domain

```text
List<FaceObservation>  (Infrastructure tạo ra, Domain không quan tâm bằng cách nào)
        │
        ▼
FaceTracker.Update(observations, now)
   • tính IoU giữa khung cũ và khung mới → ghép cặp tham lam (IoU cao ghép trước)
   • không ghép được → tạo Track mới
   • obs.Usable → đẩy vào Track.History (giữ tối đa 5)
   • Track không thấy > 1.5 s → xoá
        │  trả về [(Track, Observation)]
        ▼
foreach track: AttendanceDecider.Evaluate(track, now)
   1) LooksSpoof?  đủ 5 frame đã kiểm và TB điểm thật < 0.3  → Decision(Spoof)
   2) voters = history.Where(policy.CountsTowardVote)          ← Strategy quyết định ai được bầu
      nhóm theo EmployeeId, lấy người nhiều phiếu nhất
      ≥ 3 phiếu? → đánh dấu Done; còn trong cooldown 5' → null; không → Decision(Attendance)
   3) 5 frame hợp lệ mà không khớp ai → Decision(Unknown)
        │
        ▼
Decision? → (tầng Application sẽ lưu DB + chụp ảnh — Bài 02)
```

Vì sao phải "bỏ phiếu" qua nhiều frame? Một frame đơn lẻ có thể nhận nhầm (mờ, nghiêng). Yêu cầu 3/5 frame cùng
một người giảm mạnh nhận nhầm — đây là **quy tắc nghiệp vụ**, nên nó sống ở Domain.

---

## 3. Các kỹ thuật thiết kế trong bài

### 3.1 Strategy Pattern + OCP — `LivenessPolicies.cs`

Bản Python:

```python
def _counts(self, obs):
    if self._mode == "required":      return obs.is_real is True
    if self._mode == "opportunistic": return obs.is_real is not False
    return True
# ...và looks_spoof() lại có thêm  if self._mode == "off": return False
```

Bản .NET:

```csharp
public interface ILivenessPolicy
{
    bool CountsTowardVote(FaceObservation obs);
    bool CanFlagSpoof { get; }
}
// 3 class: LivenessOffPolicy, LivenessOpportunisticPolicy, LivenessRequiredPolicy

var voters = history.Where(policy.CountsTowardVote);  // Decider không biết đang ở chế độ nào
```

- Thêm chế độ `StrictNightPolicy` (ban đêm yêu cầu điểm cao hơn)? → **thêm 1 class**, không sửa Decider → *Open for
  extension, Closed for modification*.
- `policy.CountsTowardVote(obs)` là **runtime polymorphism**: lúc compile, C# chỉ biết kiểu `ILivenessPolicy`; lúc
  chạy CLR tra bảng hàm ảo (vtable/interface dispatch) của object thật để gọi đúng hàm.

### 3.2 Factory Method — `LivenessPolicyFactory`

```csharp
public static ILivenessPolicy Create(LivenessMode mode) => mode switch { ... };
```

Giá trị `mode` chỉ biết lúc runtime (đọc từ config). Factory gom chỗ `new` lại **một nơi duy nhất**. Đúng câu trong
bảng so sánh của bạn: *Factory trả lời "tạo ra đối tượng nào?"*.

> 💡 Câu hỏi phỏng vấn: "Strategy và Factory khác nhau thế nào khi dùng chung?"
> → Strategy là **các hành vi hoán đổi được** (3 policy). Factory là **cách chọn & tạo** ra một trong số đó.
> Chúng thường đi cặp: Factory chọn Strategy.

### 3.3 Dependency Inversion ngay trong Domain

```csharp
public sealed class AttendanceDecider(AttendanceRules rules, ILivenessPolicy policy)
```

Decider nhận `ILivenessPolicy` qua **constructor** (Constructor Injection), không tự `new`. Ở test ta truyền policy
nào cũng được; ở Bài 04, DI container sẽ truyền vào.

### 3.4 Rich Domain Model — `Employee.cs`

```csharp
public bool Active { get; private set; }     // không ai ngoài class gán được
public static Employee Create(...)           // validate ngay khi tạo → không tồn tại Employee "không tên"
public void Reactivate(string name)          // quy tắc "mã đang hoạt động thì không kích hoạt lại" nằm ở đây
```

So với "anemic model" (class chỉ có `get; set;`, logic nằm rải ở service): quy tắc nằm **cùng chỗ với dữ liệu**,
không thể bỏ qua. `private Employee() { }` là để EF Core (Bài 03) dựng object khi đọc từ DB.

### 3.5 Value Object — `FaceBox`, `FaceMatch`, `SampleEmbedding`

| Kiểu | Vì sao |
|---|---|
| `readonly record struct FaceBox` | Nhỏ (4 int), tạo hàng chục lần/giây → struct không cấp phát heap, GC nhẹ. So sánh theo giá trị. |
| `sealed record FaceMatch` | Bất biến, so sánh theo giá trị: `Assert.Equal(FaceMatch.None, m)` hoạt động ngay. |
| `sealed record FaceObservation` | Thuộc tính `required` + `init`: bắt buộc có `Box`, `DetScore`; tạo xong không sửa được. |

Entity (`Employee`) có **Id** và vòng đời → so sánh theo danh tính. Value object không có Id → so sánh theo giá trị.

### 3.6 `record ... with` — `QualityGates.cs`

```csharp
var enroll = recognitionGate with { MinFacePx = 80 };  // bản sao, chỉ đổi 1 thuộc tính
```

Đăng ký khuôn mặt cần cổng chặt hơn nhận diện. Python truyền thêm tham số `min_face_px=...`; C# dùng `with` tạo
biến thể bất biến — gọn và không có tham số "tuỳ chọn" lơ lửng.

### 3.7 Thread-safety không dùng lock — `FaceGallery.cs`

Hai luồng đụng vào gallery: luồng camera gọi `Match()` liên tục; luồng web gọi `Load()` khi thêm/xoá mẫu.

```csharp
private volatile Snapshot _snapshot;          // Snapshot là record BẤT BIẾN
public void Load(...) => _snapshot = new Snapshot(...);   // tạo mới hoàn toàn rồi tráo 1 phát
public FaceMatch Match(...) { var snap = _snapshot; ... }  // đọc tham chiếu MỘT lần
```

Gán tham chiếu là thao tác nguyên tử trong .NET → không ai thấy gallery "nửa cũ nửa mới". Gallery sẽ được đăng ký
**Singleton** (Bài 04) — đây đúng ví dụ "In-Memory cache dùng chung" trong bảng Service Lifetimes của bạn.

---

## 4. C# hiện đại dùng trong bài (đối chiếu Python)

| C# | Python tương đương | Ghi chú |
|---|---|---|
| `class AttendanceDecider(AttendanceRules rules, ...)` | `__init__(self, cfg, ...)` | *Primary constructor* (C# 12) |
| `record` / `record struct` | `@dataclass(frozen=True)` | Bất biến, Equals theo giá trị |
| `required` + `init` | field không có default | Bắt buộc gán khi tạo object |
| `int?`, `bool?` | `int \| None` | Nullable — compiler cảnh báo khi quên kiểm null |
| `obs.IsReal is not false` | `obs.is_real is not False` | Pattern matching |
| `o.Match is { EmployeeId: null }` | `o.match is not None and o.match.employee_id is None` | Property pattern |
| `GroupBy(...).MaxBy(g => g.Count())` | `Counter(...).most_common(1)` | LINQ |
| `[ ]` (collection expression) | `[]` | C# 12 |

---

## 5. Câu hỏi tự kiểm tra (phỏng vấn)

1. Vì sao `FaceTracker` và `AttendanceDecider` nằm ở **Domain** chứ không ở Application?
   <details><summary>Đáp án</summary>Chúng là quy tắc nghiệp vụ thuần (bỏ phiếu, cooldown, giả mạo), không gọi I/O. Application chỉ điều phối: lấy frame, gọi AI, gọi Domain, lưu kết quả.</details>
2. Nếu Domain cần biết "bây giờ là mấy giờ", sao không gọi `DateTimeOffset.Now` bên trong Decider?
   <details><summary>Đáp án</summary>Sẽ không test được cooldown (phải đợi 5 phút thật). Thời gian được truyền vào (`Evaluate(track, now)`); Bài 02 tiêm `TimeProvider` của .NET (qua `LocalClock`) để cung cấp `now`.</details>
3. Thêm chế độ chống giả mới cần sửa những file nào?
   <details><summary>Đáp án</summary>Thêm 1 class cài `ILivenessPolicy`, thêm 1 giá trị enum và 1 nhánh trong Factory. Decider không đổi.</details>
4. `FaceBox` là `record struct`, `FaceObservation` là `record` (class). Vì sao không cùng loại?
   <details><summary>Đáp án</summary>FaceBox nhỏ, tạo rất nhiều → struct tránh cấp phát heap. FaceObservation lớn (chứa mảng embedding 128 float) → copy struct sẽ tốn, nên dùng class (copy tham chiếu).</details>

## 6. Bài tập nhỏ

1. Viết test: hai người cùng đứng trước camera (2 khung xa nhau) → mỗi người được chấm công đúng 1 lần.
   *Gợi ý:* mỗi frame là `new[] { Obs(7, box: new(100,100,120,120)), Obs(8, box: new(400,100,120,120)) }`.
2. Viết `LivenessStrictPolicy`: chỉ bỏ phiếu khi `Liveness >= 0.8`. Thêm vào Factory, viết 1 test. Để ý: bạn không
   phải chạm vào `AttendanceDecider.cs`.

---

**Bài tiếp theo (02 — Application):** khai báo các "cổng" (`IVideoSource`, `IFaceAnalyzer`, `IEmployeeRepository`,
`IUnitOfWork`, `IFileStore`), viết `AttendancePipeline` điều phối vòng xử lý frame, và các use case
(thêm nhân viên, đăng ký khuôn mặt, xem lịch sử) theo kiểu Command/Query.
