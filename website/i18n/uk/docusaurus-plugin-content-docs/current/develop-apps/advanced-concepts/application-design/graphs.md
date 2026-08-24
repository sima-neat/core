---
title: "Розробка застосунків за допомогою графа."
description: "Як створювати моделі, вузли, іменовані вхідні/вихідні дані, гілки, об’єднувати їх та запускати за допомогою загальнодоступного API графа."
sidebar_position: 1
slug: /develop-apps/advanced-concepts/graphs
---

# Розробка застосунків за допомогою графа.

Використовуйте `Model`, коли вам потрібно лише завантажити та запустити один скомпільований архів моделі. Використовуйте `Graph`, коли ви хочете створити застосунок на основі моделей і вузлів: додавайте загальнодоступні вхідні та вихідні дані, з’єднуйте повторно використовувані фрагменти, розгалужуйте потоки, об’єднуйте потоки, перевіряйте застосунок і зберігайте або візуалізуйте те, що фактично було запущено.

Концептуальна модель навмисно проста:

| Концепція | Значення |
|---|---|
| `Model` | Завантажений з диска скомпільований архів моделі, наприклад `resnet50.tar.gz` або `yolov8.tar.gz`. |
| `Node` | Один етап обробки: вхідний, вихідний, трансформаційний, вихідний, кінцевий, модельний або допоміжний етап. |
| `Graph` | Схема взаємозв’язків компонентів застосунку: які вузли/фрагменти існують і як дані передаються між ними. |
| `Run` | Обробник, який повертається функцією `Graph::build()` під час виконання: передавання вхідних даних, отримання вихідних даних, збір метрик, зупинка. |

Коротко кажучи:

```text
Graph = what to run
Run   = the running instance
```

Почніть з перегляду сторінок із завданнями, коли вам потрібен коротший шлях:

- [Граф](/develop-apps/development-workflow/graph) навчає створенню контенту.
- [Запустіть граф.](/develop-apps/development-workflow/pipeline) навчає основам життєвого циклу середовища виконання, роботі з чергами, методам вимірювання та обчисленню пропускної здатності.
- [Вузол](/develop-apps/development-workflow/node) відображає загальні вузли та групи.
- [Тензор і зразок.](/develop-apps/development-workflow/core_types) пояснює структуру корисного навантаження та метаданих.

У більшості випадків код застосунків має використовувати загальнодоступні `simaai::neat::Graph` та `simaai::neat::Run`. Не створюйте застосунки з використанням низькорівневих просторів імен реалізації, оскільки вони не призначені для використання клієнтами як API.

## Коли мені знадобиться граф?

| Мета | Рекомендований API |
|---|---|
| Запустіть одну модель для одного набору вхідних даних. | `Model::run(...)` або `Model::build(...)` |
| Визначте межі вхідних і вихідних даних для моделі застосунку. | `Graph` |
| Створіть модель із використанням спеціальних вузлів обробки. | `Graph::add(...)` |
| Використовуйте фрагмент графа в кількох застосунках. | Повернути/передати фрагмент `Graph`. |
| Налаштуйте декілька входів або виходів. | Назва: `nodes::Input(...)` / `nodes::Output(...)` плюс `connect(...)`. |
| Розподіліть один потік даних між кількома споживачами. | `graphs::Branch(...)` |
| Об’єднайте кілька потоків в один логічний вихід. | `graphs::Combine(...)` з `CombinePolicy` |
| Збережіть або візуалізуйте виконану топологію та показники. | `save_run_json(run, ...)` |

## Перший граф: один вхід, одна модель, один вихід.

Це найпростіший повноцінний граф, що має вигляд застосунку:

```cpp
#include <neat.h>

#include <iostream>

namespace neat = simaai::neat;

int main() {
  neat::Model model("resnet50.tar.gz");

  neat::Graph app;
  app.add(neat::nodes::Input("image"));
  app.add(model);
  app.add(neat::nodes::Output("classes"));

  neat::Run run = app.build();

  neat::Tensor image = /* create or load an image tensor */;
  run.push("image", neat::TensorList{image});

  std::optional<neat::Sample> result = run.pull("classes", /*timeout_ms=*/1000);
  if (result) {
    // Consume result->tensors, result->detections, or other Sample metadata.
  }

  run.stop();
}
```

Рядок за рядком:

- `nodes::Input("image")` оголошує загальнодоступний вхідний порт під назвою `image`.
- `app.add(model)` додає вибраний маршрут моделі до графа.
- `nodes::Output("classes")` оголошує загальнодоступний вихідний порт під назвою `classes`.
- `app.build()` перевіряє та компілює весь граф і повертає `Run`.
- `run.push("image", ...)` надсилає дані до вказаного вхідного каналу.
- `run.pull("classes", ...)` отримує дані з вказаного вихідного каналу.

Така сама форма в Python:

```python
import pyneat

model = pyneat.Model("resnet50.tar.gz")

app = pyneat.Graph()
app.add(pyneat.nodes.input("image"))
app.add(model)
app.add(pyneat.nodes.output("classes"))

run = app.build()

image = ...  # Create or load a tensor-compatible object.
run.push("image", [image])

result = run.pull("classes", timeout_ms=1000)
run.stop()
```

У Python функція `Run.push(...)` очікує послідовність, що нагадує пакет даних. Передайте `[tensor]` або `[sample]`, а не окремий об’єкт тензора/зразка.

## Запуск графа

Вбудований модуль `Run` приймає ті самі типи загальнодоступних даних, які використовуються в інших частинах Neat:

| Корисне навантаження | Використовуйте, коли |
|---|---|
| `TensorList` | Ви передаєте тензори, тому додаткові метадані зразків не потрібні. |
| `Sample` | Вам потрібні мітки часу, `frame_id`, `stream_id`, метадані тексту/аудіо/відео, результати виявлення або сигнал кінця потоку (EOS). |
| `std::vector<cv::Mat>` | Вам потрібен зручний спосіб введення зображень за допомогою OpenCV. |

Найпоширеніші виклики функцій у C++:

```cpp
run.push(neat::TensorList{image});
run.push("image", neat::TensorList{image});

run.push(sample);
run.push("image", sample);

auto out = run.pull(/*timeout_ms=*/1000);
auto named = run.pull("classes", /*timeout_ms=*/1000);

neat::TensorList tensors = run.pull_tensors("classes", 1000);
neat::Sample sample_out = run.pull_samples("classes", 1000);
```

Використовуйте `pull(...)`, коли час очікування вичерпано або з’єднання закрито, і потрібно повернути порожній `std::optional`. Використовуйте `pull_tensors(...)` або `pull_samples(...)`, коли вам потрібна зручна допоміжна функція з типізацією, яка генерує виняток у разі перевищення часу очікування або виникнення помилки.

Для скінченних потоків, що надходять від програми, закрийте вхідний потік і очистіть його, перш ніж збирати остаточні показники:

```cpp
run.close_input();
while (auto out = run.pull("classes", 1000)) {
  // Drain remaining output.
}
run.stop();
```

Для отримання інформації про орієнтований на виконання завдань набір інструкцій для середовища виконання, зокрема про політику черги, визначення власника вихідних даних, відключення телеметрії та багатопотокове вимірювання, див. [Запустіть граф.](/develop-apps/development-workflow/pipeline).

## `build()` проти `build(first_input)`

Більшість графів можна створити без використання вхідних даних:

```cpp
neat::Run run = app.build();
```

Використовуйте це, коли граф уже містить достатньо інформації про форму/обмеження, або коли граф володіє своїми вихідними вузлами, наприклад, вхідними даними RTSP/файлу/статичного зображення.

Під час створення, початкова збірка надає Neat перші вхідні дані:

```cpp
neat::Run run = app.build(neat::TensorList{first_image});
```

Використовуйте це, коли перше введення має ініціювати адаптацію форми/формату перед початком потокової передачі. За замовчуванням увімкнено попередню перевірку зі збереженням початкових даних, тому Neat може один раз передати/отримати початкові дані, щоб виявити помилки на першому етапі під час збірки, замість того, щоб повернути `Run`, який одразу ж зазнає невдачі пізніше.

Для отримання даних про пропускну здатність, затримку та енергоспоживання зберігайте показники після фактичного виконання навантаження, а не одразу після збірки.

## Назви графів не є назвами кінцевих точок.

:::warning
`Graph("name")` — це позначка для діагностики, збережених файлів графа та візуалізації. Вона **не** визначає загальнодоступний вхід або вихід під назвою `name`.
:::

Неправильна ментальна модель:

```cpp
neat::Graph camera("image");
// This does not make run.push("image", ...) valid by itself.
```

Правильне оголошення кінцевої точки:

```cpp
neat::Graph camera("camera_route");
camera.add(neat::nodes::Input("image"));
```

І для отримання результату:

```cpp
neat::Graph classifier("classifier");
classifier.add(neat::nodes::Output("classes"));
```

Уявіть собі, що `Input("image")` та `Output("classes")` є своєрідними вхідними дверима фрагмента графа. Назва графа – це просто вивіска на будівлі.

## Перевіряйте назви кінцевих точок, а не намагайтеся вгадати.

Перед збіркою перевірте логічні загальнодоступні кінцеві точки, визначені графом:

```cpp
for (const auto& name : app.inputs()) {
  std::cout << "graph input: " << name << "\n";
}
for (const auto& name : app.outputs()) {
  std::cout << "graph output: " << name << "\n";
}
```

Після збірки перевірте, які саме значення приймає `Run`:

```cpp
for (const auto& name : run.input_names()) {
  std::cout << "run input: " << name << "\n";
}
for (const auto& name : run.output_names()) {
  std::cout << "run output: " << name << "\n";
}
```

Використовуйте це для визначення маршрутів моделі та будь-яких програм із кількома вхідними та вихідними даними. Відповідність кінцевим точкам відбувається точно:
`Input("image_l")` може бути пов’язаний із вхідним параметром моделі під назвою `image_l`; `Input("my_random_name")` – ні.

## Неназвані зручні API

Для графів з одним входом і одним виходом можна не вказувати назви кінцевих точок:

```cpp
neat::Graph app;
app.add(neat::nodes::Input());
app.add(model);
app.add(neat::nodes::Output());

neat::Run run = app.build();
run.push(neat::TensorList{image});
auto result = run.pull(1000);
```

Це зручно для швидких скриптів і тестів. Для більш складних застосувань краще використовувати іменовані вхідні та вихідні дані.

Якщо граф має кілька можливих входів або виходів, то неіменовані операції `push(...)` або `pull()` завершуються з помилкою та
повідомляють про доступні імена. Ця помилка є навмисною: Neat не повинен намагатися вгадати, яку саме камеру, тензор або вихідний блок ви мали на увазі.

## Моделі є фрагментами графа.

Модель `Model` можна безпосередньо додати до графа:

```cpp
neat::Model yolo("yolov8.tar.gz");

neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

`Graph::add(model)` додає обраний із архіву та параметрів моделі маршрут моделі. Цей маршрут може містити етапи попередньої обробки, виведення за допомогою MLA, подальшої обробки, перетворення тензорів і декодування для виявлення.
Вам не потрібно вручну викликати `model.graph()` для загального лінійного випадку.

Для більш складних композицій перегляньте або повторно використайте маршрут як фрагмент `Graph`:

```cpp
neat::Graph route = yolo.graph();

auto model_inputs = route.inputs();
auto model_outputs = route.outputs();
```

### Моделі з кількома входами

Для моделей із кількома вхідними даними не намагайтеся вгадати назви. Запитайте про маршрут:

```cpp
neat::Graph route = model.graph();

for (const auto& name : route.inputs()) {
  std::cout << "model expects input: " << name << "\n";
}
```

Потім дайте назви фрагментам, що передаються на вхід, щоб вони відповідали назвам вхідних даних моделі:

```cpp
neat::Graph left_camera;
left_camera.add(neat::nodes::Input("image_l"));

neat::Graph uv_camera;
uv_camera.add(neat::nodes::Input("image_uv"));

neat::Graph app;
app.connect(left_camera, route);  // Binds image_l -> model image_l.
app.connect(uv_camera, route);    // Binds image_uv -> model image_uv.
```

Якщо `left_camera` оголошено як `Input("a_new_name_image_l")`, воно не буде пов’язане з `image_l`. Замість того, щоб покладатися на неявне перейменування, додайте невеликий адаптерний граф із правильною назвою кінцевої точки.

### Окремі моделі графів

За замовчуванням, `model.graph()` повертає фрагмент моделі, який можна повторно використовувати, з відкритими іменованими кінцевими точками. Якщо ви хочете, щоб повернутий граф можна було запускати самостійно, запросіть явні загальнодоступні вхідні/вихідні вузли:

```cpp
neat::Model::RouteOptions route_opt;
route_opt.include_input = true;
route_opt.include_output = true;

neat::Graph standalone = model.graph(route_opt);
neat::Run run = standalone.build();
```

Для розширеного використання або налагодження модель маршруту може надавати доступ до окремих фізичних вихідних даних:

```cpp
route_opt.expose_all_outputs = true;
```

Залиште цю функцію вимкненою, якщо вам не потрібні окремі фізичні буфери виводу. За замовчуванням модель
показує логічний вивід моделі, який очікується відповідно до контракту маршруту. Якщо модель має лише
один фізичний вихід, `expose_all_outputs = true` все одно показує лише один вихід.

## `add()` проти `connect()`

Існує два інструменти для створення композицій:

| API | Значення | Використовуйте, коли |
|---|---|---|
| `add(x)` | Додайте або вставте в поточну лінійну послідовність. | Ви маєте на увазі «наступний етап у тому ж конвеєрі». |
| `connect(a, b)` | З’єднайте два фрагменти графа за допомогою іменованих кінцевих точок. | Ви створюєте фрагменти, які можна повторно використовувати, або розробляєте топологію. |
| `connect("a", "b")` | З’єднайте двома проводами дві кінцеві точки, які вже були визначені всередині одного й того ж графа. | Ви створюєте невеликий допоміжний фрагмент. |

Лінійна композиція:

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));
```

Склад фрагментів:

```cpp
neat::Graph app;
app.connect(camera, model_route);
app.connect(model_route, output_sink);
```

Внутрішнє підключення кінцевих точок у допоміжному фрагменті:

```cpp
neat::Graph pass_through("pass_through");
pass_through.add(neat::nodes::Input("in"));
pass_through.add(neat::nodes::Output("out"));
pass_through.connect("in", "out");
```

Основне правило: `add()` означає лінійний ланцюг. `connect()` означає топологію графа.

## Повторно використовувані фрагменти графа.

Функції можуть повертати фрагменти графа, які можна повторно використовувати:

```cpp
neat::Graph make_classifier(neat::Model& model) {
  neat::Graph g("classifier");
  g.add(neat::nodes::Input("image"));
  g.add(model);
  g.add(neat::nodes::Output("classes"));
  return g;
}
```

Використовуйте багаторазовий фрагмент лінійно:

```cpp
neat::Graph classifier = make_classifier(model);

neat::Graph app;
app.add(classifier);
```

Або явно вкажіть фрагменти дроту:

```cpp
neat::Graph app;
app.connect(camera, classifier);
app.connect(classifier, class_sink);
```

Якщо `add()` після того, як гілка стане неоднозначною, Neat не вдається і пропонує вам використовувати `connect(...)` натомість.
Це краще, ніж мовчки додавати зміни до неправильної гілки.

## Розгалуження одного потоку

Використовуйте `graphs::Branch`, коли один вхідний потік має надходити до кількох іменованих вихідних потоків:

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_input"});
```

Значення:

```text
image -> preview
      -> model_input
```

Приклад:

```cpp
neat::Graph camera;
camera.add(neat::nodes::Input("image"));

neat::Graph preview;
preview.add(neat::nodes::Output("preview"));

neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_input"});

neat::Graph app;
app.connect(camera, branch);
app.connect(branch, preview);
```

Під час підключення гілки до моделі виберіть назву вихідних даних гілки, щоб вона збігалася з назвою вхідних даних моделі:

```cpp
neat::Graph route = model.graph();
for (const auto& name : route.inputs()) {
  std::cout << "choose a branch output matching: " << name << "\n";
}
```

Розгалуження є явним, оскільки воно впливає на черги та механізми регулювання потоку даних. Якщо одна з гілок працює повільно, це може призвести до уповільнення або припинення обробки даних у порівнянні з іншою гілкою, залежно від параметрів виводу та структури графа, що обробляє дані.

Python:

```python
branch = pyneat.graphs.branch("image", ["preview", "model_input"])
```

## Об’єднання кількох потоків.

Використовуйте `graphs::Combine`, коли кілька вхідних потоків мають об’єднатися в один логічний вихід:

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "stereo",
                                         neat::CombinePolicy::ByFrame);
```

Значення:

```text
left  --\
        +--> stereo
right --/
```

Правила:

| Політика | Значення |
|---|---|
| `CombinePolicy::None` | Не об’єднуйте автоматично. Якщо до одного вихідного каналу підключено кілька джерел, система має перемикатися у закритий режим у разі їх відмови. |
| `CombinePolicy::ByFrame` | Зіставте зразки, які мають абсолютно однаковий `Sample::frame_id`. Якщо ідентифікатор кадру відсутній, зіставлення не відбудеться; механізм резервного копіювання PTS не передбачено. |
| `CombinePolicy::ByPts` | Зіставте зразки, щоб час їхньої презентації `Sample::pts_ns` був абсолютно однаковим. Відсутність PTS призводить до помилки; механізм резервного копіювання ідентифікатора кадру не передбачено. |

Проста мова:

- `ByFrame` означає: «надайте мені зразки для лівого та правого каналів з однаковим номером кадру».
- `ByPts` означає: «надайте мені зразки з однаковим часовим штампом медіафайлу».

Приклад:

```cpp
neat::Graph left;
left.add(neat::nodes::Input("left"));

neat::Graph right;
right.add(neat::nodes::Input("right"));

neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "stereo",
                                         neat::CombinePolicy::ByFrame);

neat::Graph app;
app.connect(left, pair);
app.connect(right, pair);

neat::Run run = app.build();
run.push("left", left_sample_with_frame_id_42);
run.push("right", right_sample_with_frame_id_42);
auto stereo = run.pull("stereo", 1000);
```

Python:

```python
pair = pyneat.graphs.combine(["left", "right"], "stereo", pyneat.CombinePolicy.ByFrame)
```

Якщо зразки не містять необхідного ключа, етап об’єднання завершується невдало, і замість того, щоб намагатися вгадати, виводиться діагностичне повідомлення.

## Джерела та приймачі

Існує два способи, за допомогою яких дані потрапляють у граф, і два способи, за допомогою яких вони з нього виходять.

### Вхідні дані, отримані через сповіщення застосунку.

Використовуйте `nodes::Input(...)`, коли код програми передаватиме дані:

```cpp
app.add(neat::nodes::Input("image"));
run.push("image", neat::TensorList{image});
```

### Джерело вхідних даних, що належить графу.

Використовуйте вихідний вузол або вихідний фрагмент, коли граф є власником джерела даних:

```cpp
app.add(neat::nodes::RTSPInput("rtsp://camera/stream"));
```

або фрагмент RTSP, який можна повторно використовувати після декодування:

```cpp
neat::nodes::groups::RtspDecodedInputOptions opt;
opt.url = "rtsp://camera/stream";

app.add(neat::nodes::groups::RtspDecodedInput(opt));
```

Коли граф містить власне джерело даних, зазвичай спочатку викликається функція `build()`, а потім отримуються результати; не слід передавати дані до цього джерела безпосередньо з коду програми.

### Вивід, отриманий за допомогою програми.

Використовуйте `nodes::Output(...)`, коли код програми має отримувати результати:

```cpp
app.add(neat::nodes::Output("detections"));
auto out = run.pull("detections", 1000);
```

### Вихідний вузол, що належить графу.

Використовуйте вузол або групу вихідних даних, коли потрібно, щоб граф самостійно записував результати:

```cpp
neat::UdpOutputOptions udp;
udp.host = "192.0.2.10";
udp.port = 5000;

app.add(neat::nodes::UdpOutput(udp));
```

Для граф, які створені для роботи в цьому режимі, також доступний вивід у стилі серверного протоколу RTSP:

```cpp
neat::RtspServerHandle server = app.run_rtsp(rtsp_options);
```

## Перевірка та діагностика.

Перевіряйте перед збіркою, якщо вам потрібен структурований звіт без запуску ресурсів середовища виконання:

```cpp
neat::GraphReport report = app.validate();
if (!report.error_code.empty()) {
  std::cerr << report.repro_note << "\n";
}
```

Перехоплюйте помилку `NeatError` під час викликів функцій для збірки, запуску, відправки або отримання даних:

```cpp
try {
  neat::Run run = app.build();
} catch (const neat::NeatError& e) {
  std::cerr << e.what() << "\n";

  const neat::GraphReport& report = e.report();
  std::cerr << "error_code: " << report.error_code << "\n";
  std::cerr << "hint: " << report.repro_note << "\n";
}
```

Корисні інструменти для налагодження:

```cpp
std::cout << app.describe() << "\n";
std::cout << app.describe_backend() << "\n";
```

- `describe()` виводить загальну інформацію про публічний граф: кінцеві точки, фрагменти та топологію.
- `describe_backend()` виводить детальнішу інформацію про внутрішню структуру, що може бути корисним під час налагодження створеного конвеєра.
  рядки або маршрутизація під час виконання (runtime).

Інформацію про таксономію кодів помилок і порядок їх обробки див. у розділі [Коди помилок.](/reference/error-codes/).

## Збережіть і завантажте структуру графа.

`Graph::save(path)` зберігає структуру публічного графа: вузли, назви кінцевих точок, явні ребра кінцевих точок, параметри виведення, політику об’єднання та інформацію про походження моделі та маршруту.

```cpp
app.save("app.graph.json");

neat::Graph loaded = neat::Graph::load("app.graph.json");
neat::Run run = loaded.build();
```

Це зберігає план графа, а не поточний конвеєр і не показники середовища виконання. Для отримання показників середовища виконання використовуйте експорт у формат JSON.

Важлива інформація про походження моделі та маршрут. Фрагмент моделі – це більше, ніж просто список фрагментів бекенду: він містить імена вхідних/вихідних даних, отримані з архіву моделі, параметри маршруту та метадані процесора вхідного маршруту для моделей з кількома вхідними даними. Якщо збережений граф містить фрагмент моделі, Neat зберігає шлях до архіву моделі та параметри маршруту, необхідні для його відновлення. Якщо архів відсутній під час завантаження, Neat видає інформативну помилку замість того, щоб мовчки створювати неповний маршрут.

## Експортуйте та візуалізуйте результати виконання.

`Run` знає як загальну структуру графа, так і структуру, що використовується в середовищі виконання. Його можна експортувати як артефакт JSON із зазначенням версії для використання в системах безперервної інтеграції, налагодження, для створення запитів до служби підтримки або для автономної візуалізації.

### Знімок топології, зроблений під час збірки.

Використовуйте експорт під час збирання, якщо вам потрібен артефакт одразу після створення графа:

```cpp
neat::RunOptions opt;
opt.run_export.path = "/tmp/startup.graph_run.json";
opt.run_export.label = "startup";

neat::Run run = app.build(opt);
```

Це початковий знімок топології. Він може містити нульові значення лічильників пропускної здатності/затримки, оскільки ще не було виконано жодного вимірювання.

Python:

```python
opt = pyneat.RunOptions()
opt.run_export.path = "/tmp/startup.graph_run.json"
opt.run_export.label = "startup"

run = app.build(opt)
```

### Знімок екрана після завершення запуску з відображенням показників.

Використовуйте експорт після завершення виконання завдання або його припинення:

```cpp
neat::Run run = app.build();
run.push("image", neat::TensorList{image});
auto out = run.pull("classes", 1000);

neat::RunExportOptions export_opt;
export_opt.label = "after_smoke_test";
export_opt.metadata = {{"test_name", "smoke"}};

std::string err;
if (!neat::save_run_json(run, "/tmp/final.graph_run.json", export_opt, &err)) {
  throw std::runtime_error(err);
}
```

Python:

```python
run = app.build()
run.push("image", [image])
out = run.pull("classes", timeout_ms=1000)

export_opt = pyneat.RunExportOptions()
export_opt.label = "after_smoke_test"
export_opt.metadata = {"test_name": "smoke"}

run.save_json("/tmp/final.graph_run.json", export_opt)
```

Експортер робить знімок поточної сесії; він не зупиняє її. Якщо вам потрібні остаточні дані для завершеного набору завдань, викличте `run.close_input()` і очистіть вихідні дані, або викличте `run.stop()`, перш ніж зберегти.

Щоб включити телеметрію живлення плати:

```cpp
neat::RunOptions opt;
opt.enable_board_power(/*sample_interval_ms=*/100);

neat::Run run = app.build(opt);
```

Схема JSON має версію `sima.neat.graph_run` `1`. Схема розміщена за адресою `schemas/graph_run_v1.schema.json`, а інструмент перевірки CI – за адресою `tests/perf/tools/graph_run_schema.py`.

Відобразіть артефакт без підключення до Інтернету:

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json -o /tmp/final.graph_run.html
```

Оберіть, який вигляд потрібно відобразити:

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view public
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view lowered
```

- `public` відображає граф, створений користувачем: названі вхідні та вихідні дані, фрагменти та `connect(...)`.
  краї.
- `lowered` показує, що Neat виконує всередині: сегменти конвеєра, згенеровані етапи розгалуження/об’єднання.
  черги та граничні випадки, що виникають під час роботи.

## Поширені шаблони

### Класифікація зображень

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(resnet);
app.add(neat::nodes::Output("classes"));
```

### Виявлення об’єктів

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

### Камера RTSP, модель, вивід, що отримується застосунком.

```cpp
neat::nodes::groups::RtspDecodedInputOptions source_opt;
source_opt.url = "rtsp://camera/stream";

neat::Graph app;
app.add(neat::nodes::groups::RtspDecodedInput(source_opt));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

### Вхідні дані програми для вихідного UDP-каналу, яким керує граф.

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::UdpOutput(udp_options));
```

### Перегляд гілки та шлях до моделі.

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_image"});
```

Назвіть `model_image` відповідно до вхідних даних шляху моделі або вставте явний фрагмент адаптера.

### Об’єднайте лівий і правий потоки.

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "pair",
                                         neat::CombinePolicy::ByPts);
```

Використовуйте `ByPts`, коли мітки часу медіафайлів є ключем синхронізації; використовуйте `ByFrame`, коли ідентифікатори кадрів є ключем синхронізації.

### GenAI та інші фрагменти коду.

GenAI та інші нелінійні/багатоетапні можливості все ще мають інтегруватися в код застосунку як загальнодоступні.
`Graph` фрагменти та виконувати через `Graph::build() -> Run`:

```cpp
neat::Graph app;
app.add(genai_fragment);

neat::Run run = app.build();
run.push("prompt", prompt_sample);
auto token = run.pull("tokens", 1000);
```

Точна назва фабрики фрагментів GenAI та допоміжних зразків залежить від встановленого пакета GenAI. Правило для графа залишається незмінним: додайте або з’єднайте загальнодоступні фрагменти, а потім використовуйте названі `Run::push(...)` та `Run::pull(...)`.

## Поширені помилки та підводні камені

### Не використовуйте мітки графа як кінцеві точки.

Неправильно:

```cpp
neat::Graph image("image");
run.push("image", neat::TensorList{tensor}); // Graph label is not an endpoint.
```

Правильно:

```cpp
neat::Graph image;
image.add(neat::nodes::Input("image"));
```

### Не намагайтеся вгадати назви вхідних параметрів моделі.

Неправильно:

```cpp
left.add(neat::nodes::Input("my_left"));
app.connect(left, model);
```

Правильно:

```cpp
for (const auto& name : model.graph().inputs()) {
  std::cout << name << "\n";
}
```

Потім вкажіть назви кінцевих точок вищезазначених компонентів, щоб вони відповідали.

### Не використовуйте неіменовані операції «push/pull» у графах із кількома кінцевими точками.

Неправильно:

```cpp
run.push(neat::TensorList{left});
run.push(neat::TensorList{right});
```

Правильно:

```cpp
run.push("left", neat::TensorList{left});
run.push("right", neat::TensorList{right});
```

### Не вмикайте функцію випадковим чином без налаштування CombinePolicy.

Неправильно:

```cpp
neat::Graph bundle;
bundle.add(neat::nodes::Output("bundle"));

app.connect(left, bundle);
app.connect(right, bundle); // Ambiguous: how should left/right be synchronized?
```

Правильно:

```cpp
neat::Graph bundle = neat::graphs::Combine({"left", "right"},
                                           "bundle",
                                           neat::CombinePolicy::ByFrame);
```

### Не вставляйте блоки «Вхід/Вихід» посередині, якщо тільки це не потрібно для позначення межі фрагмента.

`Input` та `Output` є публічними оголошеннями меж. У фрагментах, які можна повторно використовувати, саме це й потрібно. У суто лінійному застосунку додавання додаткового `Output` посередині може створити реальний «споживач», до якого можна передавати дані, і викликати зворотний тиск, якщо ця межа не буде використана іншим `connect(...)` з’єднанням.

### Не використовуйте низькорівневі API для роботи з графами в середовищі виконання в коді застосунку.

Уникайте використання низькорівневих API для роботи з графом у середовищі виконання під час розробки або написання коду застосунку.

Натомість використовуйте загальнодоступний інтерфейс застосунку:

```cpp
neat::Graph
neat::Run
app.build()
```

## Додаткова інформація: матеріалізація меж.

Вузли, названі `Input` і `Output`, є оголошеннями публічного інтерфейсу фрагмента. Вони мають вищий рівень, ніж об’єкти середовища виконання, які використовуються для передавання буферів.

Перед побудовою виконуваного конвеєра, `Graph::build()` нормалізує межі:

| Оголошення межі. | З’явилося, коли… | Пропущено, коли... |
|---|---|---|
| `nodes::Input("name")` | до нього не підключено жодного зовнішнього графа, тому це має бути загальнодоступна кінцева точка `Run::push("name", ...)`. | до нього надходить інформація з вищого рівня графа, тому це лише внутрішній фрагмент параметра. |
| `nodes::Output("name")` | жоден із наступних графів не використовує ці дані, тому це має бути загальнодоступна кінцева точка `Run::pull("name")`. | подальший граф використовує ці дані, тому це лише значення, яке повертається як внутрішній фрагмент. |

Те, що щось було опущено, не означає, що про це забули. Компілятор зберігає інформацію про походження, тому `describe()`, помилки перевірки, показники та експортований у форматі JSON код можуть і надалі посилатися на назву кінцевої точки, з якою взаємодіє користувач.

Це запобігає створенню повторно використовуваними фрагментами прихованих операцій введення/виведення в стилі appsrc/appsink у середині програми. Наприклад:

```cpp
neat::Graph app;
app.connect(camera, route);
app.connect(route, display);
```

Шлях до виконуваних даних — `camera -> route body -> display`, а не `camera -> route.Input -> route.Output -> display`, з додатковими фізичними приймачами/джерелами посередині.

## Короткий довідник з API

### C++

```cpp
// Composition
neat::Graph app("debug_label");
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));
app.connect(fragment_a, fragment_b);
app.connect("from_endpoint", "to_endpoint");

// Endpoint inspection
auto graph_inputs = app.inputs();
auto graph_outputs = app.outputs();

// Build/run
neat::Run run = app.build();
run.push("image", neat::TensorList{image});
auto out = run.pull("classes", 1000);

// Runtime endpoint inspection
auto run_inputs = run.input_names();
auto run_outputs = run.output_names();

// Validation/debug/export
neat::GraphReport report = app.validate();
std::cout << app.describe() << "\n";
app.save("app.graph.json");
neat::save_run_json(run, "/tmp/app.graph_run.json");
```

### Python

```python
app = pyneat.Graph("debug_label")
app.add(pyneat.nodes.input("image"))
app.add(model)
app.add(pyneat.nodes.output("classes"))

print(app.inputs())
print(app.outputs())

run = app.build()
run.push("image", [image])
out = run.pull("classes", timeout_ms=1000)

print(run.input_names())
print(run.output_names())

app.save("app.graph.json")
run.save_json("/tmp/app.graph_run.json")
```

## Для подальшого ознайомлення

- [Model programming model](/develop-apps/development-workflow/model)
- [Node programming model: groups and boundaries](/develop-apps/development-workflow/node#boundary-nodes)
- [Tensor and Sample programming model](/develop-apps/development-workflow/core_types)
- [Runtime tuning (Tutorial 016)](/tutorials/tune-throughput-and-queues)
- [Diagnostics (Tutorial 012)](/tutorials/diagnose-a-pipeline)
- [GStreamer layer](/develop-apps/advanced-concepts/gstreamer_layer)
