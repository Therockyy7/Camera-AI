# Bài 03 — Tầng Infrastructure: cài các port bằng công nghệ thật

> Project: `src/CameraAI.Infrastructure` (→ Application) · Test: `tests/CameraAI.Infrastructure.Tests` (28 test)
>
> Gói NuGet: EF Core SQLite, OpenCvSharp4, ONNX Runtime — **chỉ tầng này** có chúng.

---

## 1. Infrastructure là "adapter"

Bài 02, Application khai báo các port (interface). Bài này **cắm adapter vào từng port**:

| Port (Application) | Adapter (Infrastructure) | Công nghệ |
|---|---|---|
| `IEmployeeRepository`, `IFaceSampleRepository`, `IAttendanceRepository` | `EmployeeRepository`, … | EF Core |
| `IUnitOfWork` | `CameraAiDbContext` (chính nó) | EF Core |
| `IReportQueries` | `ReportQueries` | EF Core (LINQ chiếu ra DTO) |
| `IFileStore` | `LocalFileStore` | System.IO |
| `IVideoSource` | `OpenCvVideoSource` | OpenCvSharp `VideoCapture` |
| `IFaceAnalyzerFactory` → `IFaceAnalyzer` | `FaceAnalyzerFactory` → `TimingFaceAnalyzer(OpenCvFaceAnalyzer)` | OpenCvSharp + ONNX Runtime |
| `IFrameImaging` | `OpenCvImaging` | OpenCvSharp |

Mọi class adapter đều **`internal`**: ngoài project không ai `new` được chúng. Thứ duy nhất lộ ra là
`AddInfrastructure()` và `InitializeDatabaseAsync()`. Tầng Web vì vậy **không thể** lỡ phụ thuộc thẳng vào
`OpenCvFaceAnalyzer`; nó chỉ có thể xin `IFaceAnalyzerFactory`. Đây là cách *ép* DIP bằng trình biên dịch.

```text
src/CameraAI.Infrastructure/
├── DependencyInjection.cs        ★ AddPersistence() + AddVision() — nối port ↔ adapter
├── Persistence/
│   ├── CameraAiDbContext.cs      DbContext = Unit of Work; snake_case + converter thời gian
│   ├── EntityConfigurations.cs   ★ Fluent API: entity Domain ↔ bảng SQLite (schema của bản Python)
│   ├── Converters.cs             giờ ↔ chuỗi ISO, float[] ↔ BLOB
│   ├── Repositories.cs           3 repository (phía ghi)
│   ├── ReportQueries.cs          query phía đọc (CQRS)
│   └── DatabaseInitializer.cs    EnsureCreated + WAL
├── Files/LocalFileStore.cs       ảnh trong data/, chặn path traversal
├── Video/                        MatFrame (IVideoFrame), OpenCvVideoSource
└── Vision/
    ├── YuNetDetector.cs          phát hiện mặt
    ├── RecognitionModels.cs      ★ Abstract Factory: họ SFace / ArcFace
    ├── MiniFasLiveness.cs        chống giả
    ├── OpenCvFaceAnalyzer.cs     ráp các bước thành IFaceAnalyzer
    ├── TimingFaceAnalyzer.cs     ★ Decorator đo thời gian
    ├── FaceAnalyzerFactory.cs    tạo bộ model + bọc decorator
    ├── ImageOps.cs               căn mặt, tensor, độ nét, cắt ảnh
    └── OpenCvImaging.cs          vẽ nhãn, JPEG
```

**Thứ tự đọc gợi ý:** `DependencyInjection.cs` → `Persistence/EntityConfigurations.cs` → `Persistence/Repositories.cs`
→ `Vision/OpenCvFaceAnalyzer.cs` → `Vision/RecognitionModels.cs` → `Vision/TimingFaceAnalyzer.cs`.

---

## 2. Persistence với EF Core

### 2.1 DbContext chính là Unit of Work

```csharp
public sealed class CameraAiDbContext(...) : DbContext(options), IUnitOfWork
```

`DbContext.SaveChangesAsync(CancellationToken)` có đúng chữ ký của `IUnitOfWork.SaveChangesAsync`, nên DbContext
cài interface này **mà không cần viết thêm dòng nào**. Repository chỉ `Add` / `Remove` vào change tracker; DB chỉ
được ghi khi `SaveChangesAsync`, gói trong **một transaction**.

### 2.2 ★ Chuyển tiếp: repository và UoW phải dùng CHUNG một DbContext

Đúng điều ta đã bàn ở câu hỏi `AddSingleton<A, B>`:

```csharp
services.AddDbContext<CameraAiDbContext>(...);                                     // Scoped
services.AddScoped<IUnitOfWork>(sp => sp.GetRequiredService<CameraAiDbContext>()); // ✅ trỏ về context của scope
// services.AddScoped<IUnitOfWork, CameraAiDbContext>();                           // ❌ tạo context THỨ HAI
```

Nếu viết dòng ❌, repository sẽ `Add` vào context A còn UoW lại `Save` context B, nên **không có gì được lưu** và
cũng không có lỗi nào báo. Test `Repository_and_unit_of_work_share_one_DbContext_per_scope` khoá chặt chuyện này
bằng `Assert.Same(...)`.

### 2.3 Fluent API: Domain không biết mình được lưu ở đâu

```csharp
b.ToTable("face_samples");
b.HasOne<Employee>().WithMany().HasForeignKey(s => s.EmployeeId);
b.OwnsMany(s => s.Embeddings, e => { e.ToTable("sample_embeddings"); ... });
```

Domain không có `[Table]`, `[Column]`, `[Key]` nào. Đổi sang SQL Server thì chỉ sửa tầng này. EF đọc/ghi qua
**backing field** (`_embeddings`) và **constructor private**, nên Domain vẫn giữ được `private set`.

- **Owned type**: `SampleEmbedding` là value object (không có Id riêng), sống trong bảng riêng với khoá
  `(sample_id, model)`. Xoá mẫu thì embedding bị xoá theo.

### 2.4 ★ Dùng chung DB với bản Python

Mình map **đúng schema** của `cameraai/storage.py`: bảng/cột snake_case, `ts` dạng chuỗi
`"2026-10-26T08:00:00+07:00"`, embedding là BLOB float32. Kết quả kiểm trên bản sao `data/cameraai.db` thật:

| | sqlite3 (Python) | EF Core (.NET) |
|---|---|---|
| Nhân viên đang hoạt động | 3 | 3 |
| Embedding SFace | 12 | 12 vector × 128 chiều |
| Lần chấm công | 8 | 8, giờ đọc ra `+07:00` |

**Vì sao lưu giờ dạng chuỗi?** SQLite không có kiểu ngày giờ, và EF Core không dịch được phép so sánh `DateTimeOffset`
sang SQL cho SQLite. `LocalIsoTimeConverter` đổi mọi giờ về chuỗi ISO cùng offset `+07:00`. Khi mọi giá trị cùng
offset (Việt Nam không đổi giờ mùa hè), **so sánh chuỗi = so sánh thời gian**, nên EF dịch được `ts >= @from`: nó
tự đổi tham số qua converter rồi so chuỗi.

### 2.5 CQRS phía đọc: chiếu thẳng ra DTO

```csharp
db.Employees.AsNoTracking().Where(e => e.Active)
    .Select(e => new EmployeeDto(e.Id, e.Code, e.Name, e.CreatedAt,
        db.FaceSamples.Count(s => s.EmployeeId == e.Id)))   // → SELECT … (SELECT COUNT(*) …)
```

`AsNoTracking()`: chỉ đọc nên không bắt change tracker theo dõi, nhanh và nhẹ hơn.

### 2.6 Ba cái bẫy LINQ ta đã dính khi viết bài này (đều là lỗi thật)

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `CS1525: Invalid expression term '&&'` | Trong cú pháp query (`from … where …`), **`from` là từ khoá**, nên tham số tên `from` làm parser rối | đặt tên `start`/`end` |
| `The LINQ expression … could not be translated` | `.Select(new AttendanceRow(...)).OrderBy(r => r.Ts)`: EF không dịch được việc sắp theo thuộc tính của DTO **sau khi** đã tạo DTO | `orderby a.Timestamp` **trước** `select` |
| `CS8640: Expression tree cannot contain … Span` | lambda trong `ValueConverter`/`ValueComparer` là **expression tree** (EF đọc cấu trúc của nó), mà expression tree không chứa được `Span` | gói phần dùng Span vào hàm static rồi gọi hàm |

**EnsureCreated hay Migrations?** Dùng `EnsureCreated` vì schema đang do bản Python quản lý chung. Dự án .NET thuần
nên dùng `dotnet ef migrations add …` để có lịch sử thay đổi schema.

---

## 3. Thị giác: một frame đi qua `OpenCvFaceAnalyzer`

```text
IVideoFrame ──MatFrame.Of()──► Mat (OpenCV)
   YuNetDetector.Detect       thu nhỏ về 640 px → FaceDetectorYN → khung + 5 điểm mốc (nhân ngược tỉ lệ)
   foreach mặt:
     ĐO   : turn = RecognitionGate.TurnOffset(mắt, mắt, mũi)   ← hàm của Domain
            sharpness = ImageOps.Sharpness (phương sai Laplacian)
     HỎI  : gate.Evaluate(box, turn, sharpness, …)             ← Domain quyết "dùng được không"
     nếu dùng được:
       livenessGate.CanCheck? → MiniFasLiveness.Score         ← Domain quyết "có kiểm giả được không"
       aligner.Align → 112×112 → embedder.Embed → float[128]
   → List<FaceObservation>   (KHÔNG so gallery: pipeline tự gọi FaceGallery của Domain)
```

**Infrastructure ĐO, Domain QUYẾT.** Ngưỡng "mặt ≥ 40 px", "quay ≤ 0,45", "≥ 64 px mới kiểm giả" đều nằm trong
`RecognitionGate` / `LivenessGate` của Domain. Đổi OpenCV sang thư viện khác thì các quy tắc này không đổi.

### ★ Kiểm chứng: .NET cho ra cùng kết quả với Python

Phần AI rất dễ **sai mà không báo lỗi**. Chuẩn hoá điểm ảnh sai, nhầm BGR/RGB, căn mặt lệch: code vẫn chạy, nhưng
embedding vô nghĩa. Vì vậy mình chạy cả hai bản trên **cùng 8 ảnh mẫu** rồi so từng con số:

| Đại lượng | SFace | ArcFace |
|---|---|---|
| Khung mặt | trùng 100% | trùng 100% |
| Độ nét, độ quay | lệch 0 | lệch 0 |
| Điểm chống giả | lệch < 0,00001 | lệch < 0,00001 |
| **Cosine embedding .NET ↔ Python** | **≥ 0,999999** | **1,000000** (512 chiều) |

Bài học rút ra: với code "chuyển ngữ" từ hệ khác, **test đối chiếu (parity test)** đáng tin hơn mọi lần đọc code.

> Vì sao SFace không dùng `FaceRecognizerSF` của OpenCV như bản Python? OpenCvSharp 4.13 có `FaceDetectorYN` nhưng
> **chưa có** `FaceRecognizerSF`. Thay vào đó: tự căn mặt (phép đồng dạng 5 điểm về template ArcFace, đúng thứ
> OpenCV làm bên trong `alignCrop`) và chạy file `.onnx` bằng ONNX Runtime. Kết quả đối chiếu ở trên xác nhận
> cách làm này tương đương.

---

## 4. ★ Abstract Factory — `Vision/RecognitionModels.cs`

Muốn ra embedding đúng, cần **hai thứ khớp nhau**:

1. `IFaceAligner`: căn mặt về **đúng kích thước + template** mà model được huấn luyện.
2. `IFaceEmbedder`: chạy model với **đúng cách chuẩn hoá** điểm ảnh của model đó.

```csharp
internal interface IRecognitionModelFamily        // Abstract Factory
{
    IFaceAligner  CreateAligner();
    IFaceEmbedder CreateEmbedder(string modelsDir);
}
SFaceFamily   : aligner 112×112 + embedder (RGB, giữ 0..255)
ArcFaceFamily : aligner 112×112 + embedder (RGB, (x − 127,5) / 127,5)
```

Lắp lệch (aligner của họ này với embedder của họ kia) thì **không lỗi**, chỉ nhận nhầm người. Abstract Factory
bảo đảm **luôn lấy cả bộ từ cùng một họ**. Đúng câu trong file ôn tập: *"đảm bảo tính tương thích giữa các đối
tượng trong cùng một họ"*.

> **Thành thật về ví dụ này:** hai họ hiện có tình cờ dùng **cùng** kiểu căn 112×112, chỉ khác chuẩn hoá và file
> model. Giá trị của Abstract Factory thể hiện rõ khi thêm họ thứ ba khác hẳn (vd FaceNet cần 160×160, template
> khác): khi đó chỉ cần **thêm một class họ mới**, không sửa analyzer.

| | Factory Method (Bài 01: `LivenessPolicyFactory`) | Abstract Factory (bài này) |
|---|---|---|
| Tạo ra | **1** đối tượng (`ILivenessPolicy`) | **cả họ** đối tượng đi cùng nhau (aligner + embedder) |
| Chọn bằng | `switch` trên enum | **Keyed Services** theo chuỗi config |

### Keyed Services (.NET 8), đúng mục trong file ôn tập

```csharp
services.AddKeyedSingleton<IRecognitionModelFamily, SFaceFamily>("sface");
services.AddKeyedSingleton<IRecognitionModelFamily, ArcFaceFamily>("arcface");

var family = sp.GetKeyedService<IRecognitionModelFamily>(options.Recognition.Model)   // "sface" từ appsettings
    ?? throw new InvalidOperationException($"Không có model nhận diện '{model}'");
```

Thêm họ mới = thêm một class + một dòng `AddKeyedSingleton`. Không có `switch` nào phải sửa.

---

## 5. ★ Decorator — `TimingFaceAnalyzer`

```csharp
internal sealed class TimingFaceAnalyzer(IFaceAnalyzer inner, ILogger log) : IFaceAnalyzer
{
    public IReadOnlyList<FaceObservation> Analyze(IVideoFrame frame, RecognitionGate gate)
    {
        long started = Stopwatch.GetTimestamp();
        var result = inner.Analyze(frame, gate);      // việc thật do lớp bên trong làm
        ... cứ 200 frame ghi log thời gian trung bình
    }
}
// FaceAnalyzerFactory: return new TimingFaceAnalyzer(analyzer, logger);
```

Cùng interface với lớp bên trong, nên pipeline **không biết** mình đang gọi qua lớp bọc. Đây là ví dụ
"Performance Tracking" trong file ôn tập. Dự án giờ có **hai** decorator, ở hai tầng:

| Decorator | Bọc quanh | Thêm hành vi |
|---|---|---|
| `LoggingBehavior` (Application) | mọi Command/Query handler | log tên request + thời gian |
| `TimingFaceAnalyzer` (Infrastructure) | `IFaceAnalyzer` | đo ms/frame, số mặt/frame |

---

## 6. Một object, hai vai — camera

`OpenCvVideoSource` vừa là `IVideoSource` (pipeline đọc frame) vừa là `BackgroundService` (ASP.NET Core tự
Start khi app chạy, Stop khi Ctrl+C). Lại là bài toán **chuyển tiếp**:

```csharp
services.AddSingleton<OpenCvVideoSource>();
services.AddSingleton<IVideoSource>(sp => sp.GetRequiredService<OpenCvVideoSource>());
services.AddHostedService(sp => sp.GetRequiredService<OpenCvVideoSource>());
// ❌ services.AddHostedService<OpenCvVideoSource>() → object THỨ HAI mở webcam lần nữa → webcam bị khoá, lỗi
```

Test `Camera_is_one_object_behind_both_interfaces` kiểm bằng `Assert.Same`.

`cap.Read()` là lệnh **chặn** (đợi camera), nên vòng đọc chạy trên thread riêng
(`TaskCreationOptions.LongRunning`) để không chiếm thread pool của web.

---

## 7. Bộ nhớ native và `IDisposable`

`Mat` của OpenCV và `InferenceSession` của ONNX giữ bộ nhớ **C++**, mà GC .NET không thấy. Một frame 720p là
~2,7 MB; 8 frame/giây mà quên Dispose thì RAM tăng ~20 MB/giây, và GC không vội dọn vì "bên .NET chỉ thấy vài byte".

- `using var frame = source.TryGetLatest(...)`: trả về **bản sao** (`Mat.Clone()`), người gọi tự Dispose.
- Thread camera thay `_latest` rồi `Dispose` frame cũ **bên trong lock**. Vì `TryGetLatest` chỉ clone trong lock,
  nên không ai đang cầm frame cũ.
- `OpenCvFaceAnalyzer.Dispose()` giải phóng detector, ONNX session, model chống giả.

---

## 8. Vòng đời DI của tầng này

| Đăng ký | Vòng đời | Vì sao |
|---|---|---|
| `CameraAiDbContext`, repository, `ReportQueries` | **Scoped** | DbContext không thread-safe, mỗi request một cái |
| `IUnitOfWork` | **Scoped** (chuyển tiếp) | phải là đúng DbContext của scope |
| `IFileStore` | Singleton | không có trạng thái thay đổi |
| `OpenCvVideoSource` | Singleton + Hosted | một camera, một thread đọc |
| `IFrameImaging` | Singleton | hàm thuần |
| `IRecognitionModelFamily` | Keyed Singleton | chỉ là "công thức" tạo aligner/embedder |
| `IFaceAnalyzerFactory` | Singleton | mỗi lần `Create()` mới nạp model (pipeline và enrollment mỗi bên một bộ) |

---

## 9. Câu hỏi tự kiểm tra

1. Vì sao các class trong Infrastructure là `internal`?
   <details><summary>Đáp án</summary>Để tầng Web không thể phụ thuộc thẳng vào class cụ thể; nó chỉ xin được interface (port). Trình biên dịch ép DIP.</details>
2. `AddScoped<IUnitOfWork, CameraAiDbContext>()` sai ở đâu, khi nào mới lộ ra?
   <details><summary>Đáp án</summary>Tạo DbContext thứ hai; repository Add vào context A, UoW Save context B → không lưu gì, không lỗi. Chỉ lộ ra khi thấy dữ liệu "biến mất". Phải chuyển tiếp: `sp => sp.GetRequiredService<CameraAiDbContext>()`.</details>
3. Vì sao `ts` lưu dạng chuỗi mà vẫn so sánh lớn/nhỏ đúng?
   <details><summary>Đáp án</summary>Mọi giá trị cùng định dạng ISO và cùng offset +07:00 → thứ tự từ điển trùng thứ tự thời gian. Nếu trộn nhiều offset thì sai.</details>
4. Abstract Factory khác Factory Method thế nào, lấy ví dụ trong dự án?
   <details><summary>Đáp án</summary>Factory Method tạo 1 đối tượng (`LivenessPolicyFactory` → 1 policy). Abstract Factory tạo cả họ đối tượng phải tương thích nhau (`IRecognitionModelFamily` → aligner + embedder).</details>
5. Vì sao Analyzer chỉ "đo" mà không tự quyết mặt có dùng được không?
   <details><summary>Đáp án</summary>Quy tắc chất lượng là nghiệp vụ → nằm ở Domain (`RecognitionGate`), test được không cần OpenCV; đổi thư viện thị giác không đụng tới quy tắc.</details>

## 10. Bài tập

1. **Decorator thứ hai:** viết `CachingReportQueries : IReportQueries` bọc `ReportQueries`, cache `EmployeesAsync`
   10 giây bằng `IMemoryCache`. Đăng ký thế nào để mọi nơi xin `IReportQueries` đều nhận bản có cache? (Gợi ý:
   đăng ký `ReportQueries` riêng rồi `AddScoped<IReportQueries>(sp => new CachingReportQueries(sp.GetRequiredService<ReportQueries>(), ...))`.)
2. **Họ model mới (giả):** viết `GrayscaleSFaceFamily` (chuyển ảnh xám trước khi embed) chỉ để thử, đăng ký với khoá
   `"sface-gray"`, đổi `Recognition.Model` trong test và xem có phải sửa `OpenCvFaceAnalyzer` không.
3. **Thử bẫy chuyển tiếp:** đổi dòng `IUnitOfWork` thành `AddScoped<IUnitOfWork, CameraAiDbContext>()`, chạy
   `dotnet test` và đọc test nào đỏ, vì sao. (Nhớ hoàn tác!)

---

**Bài tiếp theo (04 — Web):** `Program.cs` làm composition root: bind `appsettings.json` → `CameraAiOptions`, gọi
`AddApplication()` + `AddInfrastructure()`, `BackgroundService` chạy pipeline, Minimal API cho 11 use case, luồng
MJPEG, JSON snake_case để dùng lại nguyên giao diện web cũ, Basic Auth, và lần đầu tiên **chạy app thật với webcam**.
