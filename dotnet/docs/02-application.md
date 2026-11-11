# Bài 02 — Tầng Application: điều phối use case

> Project: `src/CameraAI.Application` (→ Domain) · Test: `tests/CameraAI.Application.Tests` (17 test)
>
> Chạy: `cd D:\CameraAI\dotnet` rồi `dotnet test`

---

## 1. Application làm gì?

Domain biết **quyết định** ("đủ 3 phiếu thì chấm công"). Application biết **trình tự công việc** ("lấy frame → gọi
AI → so gallery → hỏi Domain → lưu DB → vẽ hình"). Nó **không tự làm việc nặng nào**: AI, DB, file, vẽ ảnh đều nhờ
người khác qua interface.

So với Domain, Application được phép biết thêm: **DI, logging, options, async/await, giao dịch**. Nó vẫn **không**
biết OpenCV, EF Core hay HTTP. Mở `CameraAI.Application.csproj`: chỉ có 3 gói `*.Abstractions` / `Options`.

```text
src/CameraAI.Application/
├── Abstractions/                ★ PORT — interface Application cần, Infrastructure cài
│   ├── Vision.cs                IVideoSource, IVideoFrame, IFaceAnalyzer(+Factory), IFrameImaging
│   ├── Persistence.cs           IEmployeeRepository, IFaceSampleRepository, IAttendanceRepository, IUnitOfWork, IFileStore
│   └── Reports.cs               IReportQueries + DTO (phía đọc CQRS), IAttendanceSink
├── Common/                      CameraAiOptions, LocalClock, exception
├── Messaging/                   ★ Mediator tự viết + LoggingBehavior (Decorator)
├── Faces/GalleryLoader.cs       nạp gallery + bảng tên từ DB
├── Pipeline/                    ★ AttendancePipeline (vòng camera), FaceLabeler, SnapshotRetention
├── Enrollment/                  FaceEnrollmentService (chụp / tải ảnh / tính lại embedding)
├── Sync/                        gửi chấm công sang hệ thống ngoài (chỗ cắm Fastdo)
├── Features/                    ★ 11 use case: Command/Query + Handler
│   ├── Employees/  Enrollment/  Reports/
└── DependencyInjection.cs       ★ AddApplication(): đăng ký + vòng đời
```

**Thứ tự đọc gợi ý:** `Abstractions/Vision.cs` → `Pipeline/AttendancePipeline.cs` → `Messaging/Mediator.cs` →
`Features/Employees/EmployeeFeatures.cs` → `DependencyInjection.cs` → `tests/.../TestHost.cs`.

---

## 2. Port (interface) — Dependency Inversion ở quy mô cả tầng

```text
             Application khai báo                      Infrastructure cài (Bài 03)
  AttendancePipeline ──► IFaceAnalyzer        ◄──── OnnxFaceAnalyzer (YuNet + SFace + MiniFASNet)
                    ──► IVideoSource          ◄──── OpenCvVideoSource (webcam / RTSP)
                    ──► IAttendanceRepository ◄──── EfAttendanceRepository (EF Core + SQLite)
                    ──► IFrameImaging         ◄──── OpenCvImaging
```

Ba quyết định thiết kế đáng nhớ:

1. **`IFaceAnalyzer` không so khớp gallery.** Bản Python để FaceEngine tự gọi `gallery.match()`. Ở đây Analyzer chỉ
   làm "ảnh → con số" (khung, điểm thật, embedding). Pipeline tự gọi `FaceGallery.Match` (Domain). → Infrastructure
   không chứa chút logic nghiệp vụ nào; đổi model AI không đụng tới phần so khớp.
2. **`IVideoFrame : IDisposable` và `TryGetLatest` trả về BẢN SAO.** Frame là bộ nhớ native của OpenCV (~2,7 MB với
   720p). GC của .NET không thấy bộ nhớ này nên không vội dọn → phải `using var frame = ...`. Trả bản sao giúp
   pipeline và màn hình đăng ký mẫu đọc cùng lúc mà không giẫm chân nhau.
3. **`IFaceAnalyzerFactory` là interface, còn `LivenessPolicyFactory` (Bài 01) là static.** Áp đúng 3 câu hỏi ta đã
   bàn: class nghiệp vụ (pipeline, enrollment) gọi nó ✔, test cần bản giả không nạp model ✔, bản thật cần
   `IOptions` + nạp file ONNX ✔ → interface.

---

## 3. Luồng 1 — Một frame camera đi qua `AttendancePipeline`

```text
BackgroundService (Web, Bài 04) ──► pipeline.RunAsync(ct)
   PrimeAsync: mở scope → LastTimesAsync (khôi phục cooldown) → GalleryLoader.ReloadAsync
   loop ~8 lần/giây:
     using frame = source.TryGetLatest(lastId)             ← port
     ProcessFrameAsync(frame, now):
        obs   = analyzer.Analyze(frame, gate)              ← port (AI)
        obs   = obs with { Match = gallery.Match(emb) }    ← Domain
        pairs = tracker.Update(obs, now)                   ← Domain
        foreach track: decider.Evaluate(track, now)        ← Domain
             └─ có Decision → RecordAsync:
                  mở scope NGẮN → IFileStore.Save(ảnh) → repo.Add(event) → uow.SaveChanges → đóng scope
        labels = labeler.For(track, obs)                   ← Application (chọn chữ/màu)
        _latest = imaging.EncodeAnnotated(frame, labels)   ← port → web đọc để phát MJPEG
     mỗi ngày 1 lần: SnapshotRetention.CleanupAsync
```

So với `pipeline.py`: logic giống hệt. Khác biệt duy nhất là **mọi thứ bên ngoài đều qua interface**.

---

## 4. ★ Vòng đời DI và bẫy Captive Dependency

Đây là phần thực chiến nhất của bài, áp đúng bảng *Service Lifetimes* trong file ôn tập. Mở `DependencyInjection.cs`:

| Vòng đời | Class | Vì sao |
|---|---|---|
| **Singleton** | `FaceGallery`, `EmployeeNames` | cache trong RAM, dùng chung mọi luồng |
| **Singleton** | `AttendancePipeline` | một vòng lặp duy nhất, giữ trạng thái tracker xuyên suốt |
| **Singleton** | `AttendanceDecider`, `ILivenessPolicy` | giữ "lần chấm gần nhất" (cooldown) |
| **Singleton** | `EnrollmentAnalyzer` | model AI nạp 1 lần (tốn RAM, chậm) + khoá dùng chung |
| **Scoped** | `GalleryLoader`, `FaceEnrollmentService`, handler, `ISender` | cần repository → cần DbContext |
| **Scoped** | *(Bài 03)* `DbContext`, repository | 1 DbContext / 1 request — DbContext **không** thread-safe |

### Bẫy: Singleton giữ Scoped

`AttendancePipeline` (Singleton) cần ghi DB. Cách viết **SAI** mà ai cũng nghĩ tới đầu tiên:

```csharp
public AttendancePipeline(IAttendanceRepository repo, ...)  // ❌ repo là Scoped
```

Singleton sống mãi → repo và DbContext bên trong cũng bị "bắt cóc" sống mãi (**captive dependency**):
- DbContext tích luỹ mọi entity đã đọc (change tracker) → rò bộ nhớ, dữ liệu cũ.
- Luồng camera và luồng web có thể dùng **cùng một DbContext** cùng lúc → lỗi
  *"A second operation was started on this context"*.

Cách **ĐÚNG** (đang dùng trong code):

```csharp
public AttendancePipeline(IServiceScopeFactory scopes, ...)   // ✅ Singleton giữ "nhà máy tạo scope"
...
await using (var scope = _scopes.CreateAsyncScope())           // mỗi lần ghi: 1 scope = 1 DbContext mới
{
    var repo = scope.ServiceProvider.GetRequiredService<IAttendanceRepository>();
    ...
}                                                              // đóng scope → DbContext được Dispose
```

**Quy tắc nhớ:** dài sống được phép giữ ngắn sống? **Không.** Ngắn sống giữ dài sống? **Được.**
`GalleryLoader` (Scoped) giữ `FaceGallery` (Singleton) là hoàn toàn hợp lệ.

> 🧪 Test bắt lỗi này tự động: `TestHost` dựng container với `ValidateScopes = true, ValidateOnBuild = true`
> (giống môi trường Development của ASP.NET Core). Thử đổi constructor pipeline sang nhận `IAttendanceRepository`
> rồi chạy `dotnet test`: container sẽ báo lỗi ngay lúc dựng.

---

## 5. ★ Mediator + Decorator — `Messaging/Mediator.cs`

### Luồng 2 — Một HTTP request (Bài 04 sẽ nối endpoint)

```text
POST /api/employees  {code, name}
  endpoint:  sender.Send(new CreateEmployeeCommand(code, name))      ← endpoint CHỈ biết ISender
     Sender: tìm IRequestHandler<CreateEmployeeCommand, EmployeeDto> trong DI
             bọc bằng các IPipelineBehavior (Decorator)
        LoggingBehavior ──► CreateEmployeeHandler.Handle()
                               repo.FindByCodeAsync → Employee.Create / Reactivate (Domain validate)
                               uow.SaveChangesAsync
                            ◄── EmployeeDto
        LoggingBehavior ghi "CreateEmployeeCommand: 3 ms"
```

**Mediator** giải quyết gì? Không có nó, mỗi endpoint phải tiêm đúng service nó cần, và service phình dần thành
"EmployeeService" có 15 method. Có Mediator: mỗi use case là **một cặp (Request, Handler)** riêng, đúng SRP. Thêm
use case mới = thêm 1 file. `RegisterHandlers` quét assembly tự đăng ký, không sửa gì khác (OCP).

**Pipeline behavior = Decorator.** Mỗi behavior bọc lớp bên trong như củ hành:

```text
   Send ─► [ A: Logging ─► [ B: Validation ─► [ Handler ] ] ]
   log:     A>              B>                 handler   <B   <A
```

Test `MediatorTests` kiểm đúng thứ tự này. Muốn thêm validation hay transaction cho **cả 11 use case**? Viết 1
behavior, đăng ký 1 dòng, không đụng handler nào. Đây đúng câu *"bổ sung hành vi mà không sửa mã nguồn class ban
đầu"* trong file ôn tập.

> Vì sao tự viết thay vì dùng MediatR? MediatR từ v13 chuyển sang giấy phép thương mại. Tự viết ~70 dòng vừa đủ,
> lại thấy rõ phép màu bên trong: chỉ là `GetRequiredService<IRequestHandler<TReq,TRes>>()` cộng một vòng `foreach`
> bọc behavior. Các interface cố ý đặt tên giống MediatR (`IRequest`, `IRequestHandler`, `ISender`,
> `IPipelineBehavior`) → sau này đổi qua MediatR gần như chỉ đổi `using`.

---

## 6. CQRS, Repository và Unit of Work

| | Phía GHI (Command) | Phía ĐỌC (Query) |
|---|---|---|
| Ví dụ | `CreateEmployeeCommand`, `DeleteSampleCommand` | `GetAttendanceDayQuery`, `ListEmployeesQuery` |
| Đi qua | **Entity** của Domain (giữ quy tắc) | **DTO** phẳng, chiếu thẳng từ DB |
| Interface | `I...Repository` + `IUnitOfWork` | `IReportQueries` |
| Vì sao | `Employee.Reactivate()` phải được gọi, không ai được `UPDATE` bậy | màn hình lịch sử cần JOIN 2 bảng + GROUP BY; dựng Entity rồi map lại là thừa |

**Repository không có `Save()`**, chỉ `Add / Remove / Get`. Việc chốt giao dịch là của `IUnitOfWork`:

```csharp
employee.Deactivate();
foreach (var s in faces) samples.Remove(s);   // 2 repository khác nhau...
await uow.SaveChangesAsync(ct);               // ...MỘT giao dịch: hoặc cả hai, hoặc không gì cả
```

Bài 03 sẽ thấy: cả repository lẫn UoW đều dùng **chung một DbContext Scoped**, và `IUnitOfWork.SaveChangesAsync`
chính là `DbContext.SaveChangesAsync`.

DTO đặt tên PascalCase (`SampleCount`, `FirstTs`). Bài 04 bật JSON snake_case → `sample_count`, `first_ts` khớp
giao diện web cũ, không sửa dòng JS nào.

---

## 7. Thời gian: `TimeProvider` + `LocalClock`

Trong Bài 01, Domain nhận `now` qua tham số. Ở Application, ai cung cấp `now`? Là `LocalClock`, dựa trên
**`TimeProvider`** có sẵn từ .NET 8 (đúng ví dụ "static `DateTimeOffset.Now` gây hại" ta đã bàn):

```csharp
services.TryAddSingleton(TimeProvider.System);          // chạy thật
services.AddSingleton<TimeProvider>(new FakeTimeProvider(...));  // test: tự tua giờ
```

Test `Capture_without_camera_reports_no_camera` đặt `AutoAdvanceAmount = 1s`: mỗi lần đọc đồng hồ, thời gian trôi
1 giây. Vòng chờ 15 giây kết thúc **trong vài mili-giây**, không phải ngồi đợi thật.

`LocalClock` còn lo múi giờ: `DayRange(26/10)` trả về `[26/10 00:00+07:00, 27/10 00:00+07:00)`. "Hôm nay" là
hôm nay ở **Việt Nam**, kể cả khi máy chủ đặt giờ UTC.

**Options → record Domain:** `CameraAiOptions` (chuỗi/số từ appsettings) có các hàm `ToGate()`, `ToRules()` đổi
sang `RecognitionGate`, `AttendanceRules` của Domain → Domain không bao giờ thấy `IOptions`.

---

## 8. Test bằng fake — tác dụng thật của port

`tests/CameraAI.Application.Tests/Fakes.cs`: mỗi port có một bản giả vài chục dòng (`FakeVideoSource`,
`FakeAnalyzer`, `InMemoryDb`...). `TestHost` dựng **DI container thật** bằng `AddApplication()` y như Program.cs,
chỉ thay port bằng fake. Vì vậy test kiểm được cả 3 thứ: logic, thứ tự điều phối, **và** việc đăng ký DI.

```csharp
host.Analyzer.Faces = (_, _) => [Face(Vectors.Unit(0))];  // "camera thấy An"
for (int i = 0; i < 5; i++) await pipeline.ProcessFrameAsync(new FakeFrame(i + 1), now.AddSeconds(i * 0.125));
Assert.Single(host.Db.Attendance);                         // chấm đúng 1 lần, có ảnh, nhãn xanh "An - đã chấm 08:00"
```

Không camera, không model 30 MB, không SQLite: toàn bộ luồng chấm công chạy dưới 1 giây.

---

## 9. Câu hỏi tự kiểm tra

1. Vì sao `GalleryLoader` là Scoped, còn `FaceGallery` là Singleton?
   <details><summary>Đáp án</summary>GalleryLoader cần repository (Scoped, giữ DbContext). FaceGallery là cache trong RAM dùng chung. Scoped giữ Singleton là hợp lệ.</details>
2. Nếu bỏ `IUnitOfWork` và cho mỗi repository tự `SaveChanges()`, xoá nhân viên có thể hỏng thế nào?
   <details><summary>Đáp án</summary>Nhân viên đã bị đánh dấu ngừng nhưng xoá mẫu lỗi giữa chừng → còn mẫu khuôn mặt của người đã nghỉ (vi phạm yêu cầu xoá dữ liệu sinh trắc). UoW gom cả hai vào một transaction.</details>
3. Trong `SnapshotRetention`, vì sao xoá file **sau** `SaveChangesAsync`, không phải trước?
   <details><summary>Đáp án</summary>Nếu DB lỗi, bản ghi vẫn trỏ tới ảnh và ảnh vẫn còn. Làm ngược lại thì DB lỗi sẽ để lại bản ghi trỏ tới file đã mất.</details>
4. `LoggingBehavior` là pattern gì? Thêm "đo thời gian" cho 11 use case mà không sửa handler nào là đúng nguyên lý nào?
   <details><summary>Đáp án</summary>Decorator; Open/Closed Principle.</details>
5. Handler `CreateEmployeeHandler` không tự kiểm "mã rỗng". Ai kiểm, và vì sao đặt ở đó?
   <details><summary>Đáp án</summary>`Employee.Create` (Domain). Quy tắc nằm cùng dữ liệu → mọi đường tạo Employee (API, import, test) đều bị kiểm, không ai quên được.</details>

## 10. Bài tập

1. **Behavior mới:** viết `SlowRequestAlertBehavior` hoặc `ValidationBehavior<TReq,TRes>` (vd: chặn `Code` dài
   quá 20 ký tự cho `CreateEmployeeCommand`). Đăng ký trong `AddApplication`, viết test giống `MediatorTests`.
2. **Thử bẫy captive dependency:** đổi constructor `AttendancePipeline` nhận `IAttendanceRepository`, chạy
   `dotnet test`, đọc thông báo lỗi. (Nhớ hoàn tác!)
3. **Use case mới:** `RenameEmployeeCommand(int Id, string Name)`. Cần thêm method gì ở `Employee` (Domain)? Có phải
   sửa `DependencyInjection.cs` không?

---

**Bài tiếp theo (03 — Infrastructure):** cài các port bằng công nghệ thật: OpenCvSharp (camera, YuNet, SFace),
ONNX Runtime (MiniFASNet, ArcFace), EF Core + SQLite (repository, UoW, query), file store, và **Abstract Factory**
cho bộ model nhận diện (SFace/ArcFace).
