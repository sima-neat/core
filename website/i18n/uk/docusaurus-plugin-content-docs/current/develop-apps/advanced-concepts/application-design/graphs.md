---
title: "Створення застосунків за допомогою Graph"
description: "Як компонувати моделі, вузли, іменовані входи й виходи, гілки, об’єднання та запуски за допомогою публічного API Graph."
sidebar_position: 1
slug: /develop-apps/advanced-concepts/graphs
---

# Створення застосунків за допомогою Graph

Використовуйте `Model`, коли потрібно лише завантажити й запустити один скомпільований архів моделі. Використовуйте `Graph`, коли потрібно побудувати застосунок навколо моделей і вузлів: додати публічні входи й виходи, з’єднати повторно використовувані фрагменти, розгалузити потоки, об’єднати потоки, перевірити застосунок і зберегти або візуалізувати те, що фактично виконувалося.

Ментальна модель навмисно проста:

| Поняття | Значення |
|---|---|
| `Model` | Скомпільований архів моделі, завантажений із диска, наприклад `resnet50.tar.gz` або `yolov8.tar.gz`. |
| `Node` | Один крок обробки: вхід, вихід, перетворення, джерело, приймач, етап моделі або допоміжний етап. |
| `Graph` | План з’єднань застосунку: які вузли й фрагменти існують і як дані проходять між ними. |
| `Run` | Дескриптор активного виконання, який повертає `Graph::build()`: через нього надсилають входи, отримують виходи, збирають метрики й зупиняють виконання. |

Коротко:

```text
Graph = what to run
Run   = the running instance
```

Якщо потрібен коротший шлях, почніть зі сторінок, присвячених завданням:

- [Граф](/develop-apps/development-workflow/graph) пояснює, як створювати графи.
- [Запуск графа](/develop-apps/development-workflow/pipeline) пояснює життєвий цикл середовища виконання, черги, вимірювання та пропускну здатність.
- [Вузол](/develop-apps/development-workflow/node) містить огляд поширених вузлів і груп.
- [Тензор і зразок](/develop-apps/development-workflow/core_types) пояснює корисні дані й метадані.

Більшість коду застосунків має використовувати публічні `simaai::neat::Graph` і `simaai::neat::Run`. Не будуйте застосунки на низькорівневих просторах імен реалізації: це не API для клієнтів.

## Коли потрібен Graph?

| Мета | Рекомендований API |
|---|---|
| Запустити одну модель на одному вході | `Model::run(...)` або `Model::build(...)` |
| Додати межі входу й виходу застосунку навколо моделі | `Graph` |
| Скомпонувати модель із власними вузлами обробки | `Graph::add(...)` |
| Повторно використати фрагмент графа в кількох застосунках | Повертати/передавати фрагмент `Graph` |
| Маршрутизувати кілька входів або виходів | Іменовані `nodes::Input(...)` / `nodes::Output(...)` плюс `connect(...)` |
| Розгалузити один потік на кількох споживачів | `graphs::Branch(...)` |
| Об’єднати кілька потоків в один логічний вихід | `graphs::Combine(...)` з `CombinePolicy` |
| Зберегти або візуалізувати виконану топологію та метрики | `save_run_json(run, ...)` |

## Перший граф: один вхід, одна модель, один вихід

Це найменший повний граф у стилі застосунку:

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

- `nodes::Input("image")` оголошує публічні вхідні двері з назвою `image`.
- `app.add(model)` вставляє в граф вибраний маршрут моделі.
- `nodes::Output("classes")` оголошує публічні вихідні двері з назвою `classes`.
- `app.build()` перевіряє й компілює весь граф і повертає `Run`.
- `run.push("image", ...)` надсилає дані в іменований вхід.
- `run.pull("classes", ...)` отримує дані з іменованого виходу.

Та сама структура в Python:

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

У Python `Run.push(...)` очікує послідовність, подібну до пакета (batch). Передавайте `[tensor]` або `[sample]`, а не окремий об’єкт тензора чи зразка.

## Запуск графа

Побудований `Run` приймає ті самі публічні типи корисних даних, що й інші частини Neat:

| Корисні дані | Коли використовувати |
|---|---|
| `TensorList` | Ви передаєте тензори, і додаткові метадані зразка не потрібні. |
| `Sample` | Потрібні мітки часу, `frame_id`, `stream_id`, метадані тексту/аудіо/відео, результати виявлення або EOS. |
| `std::vector<cv::Mat>` | Потрібен зручний вхід зображень з OpenCV. |

Поширені виклики в C++:

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

Використовуйте `pull(...)`, коли в разі тайм-ауту чи закриття потрібно отримати порожній `std::optional`. Використовуйте `pull_tensors(...)` або `pull_samples(...)`, коли потрібна типізована допоміжна функція, яка в разі тайм-ауту чи помилки генерує виняток.

Для скінченних потоків, у які дані надсилає застосунок, закрийте вхід і спорожніть вихід, перш ніж збирати остаточні метрики:

```cpp
run.close_input();
while (auto out = run.pull("classes", 1000)) {
  // Drain remaining output.
}
run.stop();
```

Практичні рекомендації щодо середовища виконання, зокрема про політику черг, володіння вихідними даними, телеметрію відкинутих даних і вимірювання для кількох потоків, див. на сторінці [Запуск графа](/develop-apps/development-workflow/pipeline).

## `build()` проти `build(first_input)`

Більшість графів можна побудувати без вхідного зразка:

```cpp
neat::Run run = app.build();
```

Використовуйте цей варіант, коли граф уже оголошує достатньо інформації про форму й caps або коли граф володіє своїми вузлами-джерелами, як-от входами RTSP, файлу чи нерухомого зображення.

Побудова з початковим входом (seeded build) передає Neat перший вхід під час побудови:

```cpp
neat::Run run = app.build(neat::TensorList{first_image});
```

Використовуйте цей варіант, коли перший вхід має задати адаптацію форми чи формату до початку потокової обробки. Попередню перевірку під час побудови з початковим входом увімкнено за замовчуванням, тому Neat може один раз надіслати й отримати цей вхід (push/pull), щоб виявити збої першого зразка ще під час побудови, а не повертати `Run`, який згодом одразу ж зазнає збою.

Щоб отримати показники пропускної здатності, затримки та енергоспоживання, зберігайте метрики після виконання реального навантаження, а не одразу після побудови.

## Назви графів — не назви кінцевих точок

:::warning
`Graph("name")` — це мітка для діагностики, збережених файлів графа й візуалізації. Вона **не** оголошує публічний вхід чи вихід з назвою `name`.
:::

Хибна ментальна модель:

```cpp
neat::Graph camera("image");
// This does not make run.push("image", ...) valid by itself.
```

Правильне оголошення кінцевої точки:

```cpp
neat::Graph camera("camera_route");
camera.add(neat::nodes::Input("image"));
```

І для виходу:

```cpp
neat::Graph classifier("classifier");
classifier.add(neat::nodes::Output("classes"));
```

Вважайте `Input("image")` і `Output("classes")` публічними дверима фрагмента графа. Назва графа — лише вивіска на будівлі.

## Перевірка назв кінцевих точок замість здогадок

Перед побудовою перегляньте логічні публічні кінцеві точки, оголошені графом:

```cpp
for (const auto& name : app.inputs()) {
  std::cout << "graph input: " << name << "\n";
}
for (const auto& name : app.outputs()) {
  std::cout << "graph output: " << name << "\n";
}
```

Після побудови перегляньте, що саме приймає `Run`:

```cpp
for (const auto& name : run.input_names()) {
  std::cout << "run input: " << name << "\n";
}
for (const auto& name : run.output_names()) {
  std::cout << "run output: " << name << "\n";
}
```

Робіть це для маршрутів моделей і будь-якого застосунку з кількома входами чи виходами. Кінцеві точки зіставляються точно:
`Input("image_l")` може прив’язатися до входу моделі з назвою `image_l`, а `Input("my_random_name")` — ні.

## Зручні API без назв

Для графів з одним входом і одним виходом назви кінцевих точок можна не вказувати:

```cpp
neat::Graph app;
app.add(neat::nodes::Input());
app.add(model);
app.add(neat::nodes::Output());

neat::Run run = app.build();
run.push(neat::TensorList{image});
auto result = run.pull(1000);
```

Це зручно для швидких скриптів і тестів. У нетривіальних застосунках краще використовувати назви.

Якщо граф має кілька можливих входів або виходів, `push(...)` чи `pull()` без назви завершується відмовою (fail-closed) і повідомляє доступні назви. Ця відмова навмисна: Neat не повинен вгадувати, яку камеру, тензор чи вихід моделі ви мали на увазі.

## Моделі — це фрагменти графа

`Model` можна додати безпосередньо до графа:

```cpp
neat::Model yolo("yolov8.tar.gz");

neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

`Graph::add(model)` вставляє маршрут моделі, вибраний на основі архіву й параметрів моделі. Цей маршрут може містити етапи попередньої обробки, інференсу на MLA, постобробки, перетворення тензорів і декодування результатів виявлення.
У типовому лінійному випадку викликати `model.graph()` вручну не потрібно.

Для складнішої композиції перегляньте або повторно використайте маршрут як фрагмент `Graph`:

```cpp
neat::Graph route = yolo.graph();

auto model_inputs = route.inputs();
auto model_outputs = route.outputs();
```

### Моделі з кількома входами

Для моделей із кількома входами не вгадуйте назви. Запитайте їх у маршруту:

```cpp
neat::Graph route = model.graph();

for (const auto& name : route.inputs()) {
  std::cout << "model expects input: " << name << "\n";
}
```

Потім назвіть фрагменти вище за потоком відповідно до назв входів моделі:

```cpp
neat::Graph left_camera;
left_camera.add(neat::nodes::Input("image_l"));

neat::Graph uv_camera;
uv_camera.add(neat::nodes::Input("image_uv"));

neat::Graph app;
app.connect(left_camera, route);  // Binds image_l -> model image_l.
app.connect(uv_camera, route);    // Binds image_uv -> model image_uv.
```

Якби `left_camera` оголошував `Input("a_new_name_image_l")`, він не прив’язався б до `image_l`. Замість того щоб покладатися на неявне перейменування, додайте невеликий граф-адаптер із правильною назвою кінцевої точки.

### Самостійні графи моделей

За замовчуванням `model.graph()` повертає повторно використовуваний фрагмент моделі з відкритими іменованими кінцевими точками. Якщо потрібно, щоб повернутий граф можна було запускати самостійно, запросіть явні публічні вузли входу й виходу:

```cpp
neat::Model::RouteOptions route_opt;
route_opt.include_input = true;
route_opt.include_output = true;

neat::Graph standalone = model.graph(route_opt);
neat::Run run = standalone.build();
```

Для складніших сценаріїв або налагодження маршрут моделі може відкривати окремі фізичні виходи:

```cpp
route_opt.expose_all_outputs = true;
```

Залишайте цей параметр вимкненим, якщо вам не потрібні саме окремі фізичні вихідні буфери. За замовчуванням модель
відкриває логічний вихід моделі, якого очікує контракт маршруту. Якщо модель має лише
один фізичний вихід, `expose_all_outputs = true` однаково відкриває лише один вихід.

## `add()` проти `connect()`

Є два інструменти композиції:

| API | Значення | Коли використовувати |
|---|---|---|
| `add(x)` | Додати в кінець поточного лінійного ланцюга або вставити в нього. | Ви маєте на увазі «наступний крок у тому самому конвеєрі». |
| `connect(a, b)` | З’єднати два фрагменти графа через іменовані кінцеві точки. | Ви компонуєте повторно використовувані фрагменти або будуєте топологію. |
| `connect("a", "b")` | З’єднати дві кінцеві точки, уже оголошені в тому самому графі. | Ви будуєте невеликий допоміжний фрагмент. |

Лінійна композиція:

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));
```

Композиція фрагментів:

```cpp
neat::Graph app;
app.connect(camera, model_route);
app.connect(model_route, output_sink);
```

Внутрішнє з’єднання кінцевих точок у допоміжному фрагменті:

```cpp
neat::Graph pass_through("pass_through");
pass_through.add(neat::nodes::Input("in"));
pass_through.add(neat::nodes::Output("out"));
pass_through.connect("in", "out");
```

Головне правило: `add()` означає лінійний ланцюг. `connect()` означає топологію графа.

## Повторно використовувані фрагменти графа

Функції можуть повертати повторно використовувані фрагменти графа:

```cpp
neat::Graph make_classifier(neat::Model& model) {
  neat::Graph g("classifier");
  g.add(neat::nodes::Input("image"));
  g.add(model);
  g.add(neat::nodes::Output("classes"));
  return g;
}
```

Використовуйте такий фрагмент лінійно:

```cpp
neat::Graph classifier = make_classifier(model);

neat::Graph app;
app.add(classifier);
```

Або з’єднайте фрагменти явно:

```cpp
neat::Graph app;
app.connect(camera, classifier);
app.connect(classifier, class_sink);
```

Якщо `add()` після розгалуження був би неоднозначним, Neat завершується помилкою й підказує натомість використати `connect(...)`.
Це краще, ніж непомітно додати крок не до тієї гілки.

## Розгалуження одного потоку

Використовуйте `graphs::Branch`, коли один вхідний потік має надходити до кількох іменованих виходів:

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

Під’єднуючи гілку до моделі, виберіть назву виходу гілки так, щоб вона збігалася з назвою входу моделі:

```cpp
neat::Graph route = model.graph();
for (const auto& name : route.inputs()) {
  std::cout << "choose a branch output matching: " << name << "\n";
}
```

Розгалуження задається явно, бо воно впливає на черги й зворотний тиск. Якщо одна гілка повільна, вона може сповільнюватися або відкидати дані відносно іншої гілки — залежно від параметрів виходу та графа нижче за потоком.

Python:

```python
branch = pyneat.graphs.branch("image", ["preview", "model_input"])
```

## Об’єднання кількох потоків

Використовуйте `graphs::Combine`, коли кілька вхідних потоків мають стати одним логічним виходом:

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

Політики:

| Політика | Значення |
|---|---|
| `CombinePolicy::None` | Не об’єднувати автоматично. Кілька виробників, під’єднаних до одного виходу, спричиняють відмову (fail-closed). |
| `CombinePolicy::ByFrame` | Зіставляти зразки з точно однаковим `Sample::frame_id`. Відсутній ідентифікатор кадру спричиняє помилку; резервного переходу на PTS немає. |
| `CombinePolicy::ByPts` | Зіставляти зразки з точно однаковою міткою часу показу `Sample::pts_ns`. Відсутній PTS спричиняє помилку; резервного переходу на ідентифікатор кадру немає. |

Простими словами:

- `ByFrame` означає: «дай мені лівий і правий зразки з однаковим номером кадру».
- `ByPts` означає: «дай мені зразки з однаковою медіаміткою часу».

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

Якщо зразки не містять потрібного ключа, етап об’єднання завершується помилкою з діагностичним повідомленням, а не вгадує.

## Джерела та приймачі

Дані потрапляють у граф двома способами й виходять із нього також двома.

### Вхід, у який дані надсилає застосунок

Використовуйте `nodes::Input(...)`, коли дані надсилатиме код застосунку:

```cpp
app.add(neat::nodes::Input("image"));
run.push("image", neat::TensorList{image});
```

### Джерело входу, яким володіє граф

Використовуйте вузол-джерело або фрагмент-джерело, коли граф володіє джерелом даних:

```cpp
app.add(neat::nodes::RTSPInput("rtsp://camera/stream"));
```

або повторно використовуваний фрагмент декодованого RTSP:

```cpp
neat::nodes::groups::RtspDecodedInputOptions opt;
opt.url = "rtsp://camera/stream";

app.add(neat::nodes::groups::RtspDecodedInput(opt));
```

Коли граф володіє своїм джерелом, ви зазвичай викликаєте `build()`, а потім отримуєте виходи (pull); надсилати дані (push) у це джерело з коду застосунку не потрібно.

### Вихід, з якого дані отримує застосунок

Використовуйте `nodes::Output(...)`, коли результати має отримувати код застосунку:

```cpp
app.add(neat::nodes::Output("detections"));
auto out = run.pull("detections", 1000);
```

### Вихідний приймач, яким володіє граф

Використовуйте вузол або групу вихідного приймача, коли граф має сам записувати результати:

```cpp
neat::UdpOutputOptions udp;
udp.host = "192.0.2.10";
udp.port = 5000;

app.add(neat::nodes::UdpOutput(udp));
```

Для графів, побудованих для такого режиму, також доступний вихід RTSP у режимі сервера:

```cpp
neat::RtspServerHandle server = app.run_rtsp(rtsp_options);
```

## Перевірка та діагностика

Виконайте перевірку до побудови, якщо потрібен структурований звіт без запуску ресурсів середовища виконання:

```cpp
neat::GraphReport report = app.validate();
if (!report.error_code.empty()) {
  std::cerr << report.repro_note << "\n";
}
```

Перехоплюйте `NeatError` навколо викликів build/run/push/pull:

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

Корисні допоміжні засоби для налагодження:

```cpp
std::cout << app.describe() << "\n";
std::cout << app.describe_backend() << "\n";
```

- `describe()` виводить зведення публічного графа: кінцеві точки, фрагменти й топологію.
- `describe_backend()` виводить низькорівневі деталі бекенду, корисні під час налагодження згенерованих рядків конвеєра
  або маршрутизації в середовищі виконання.

Таксономію кодів помилок і порядок розбору помилок див. на сторінці [Каталог кодів помилок](/reference/error-codes/).

## Збереження та завантаження композиції Graph

`Graph::save(path)` записує публічну композицію графа: вузли, назви кінцевих точок, явні ребра між кінцевими точками, параметри виходів, політику об’єднання та походження маршрутів моделей.

```cpp
app.save("app.graph.json");

neat::Graph loaded = neat::Graph::load("app.graph.json");
neat::Run run = loaded.build();
```

Так зберігається план графа, а не запущений конвеєр і не метрики середовища виконання. Для метрик середовища виконання використовуйте експорт Run у JSON.

Походження маршруту моделі має значення. Фрагмент моделі — це більше, ніж набір фрагментів коду бекенду: він містить назви входів і виходів, отримані з архіву моделі, параметри маршруту та, для моделей із кількома входами, метадані процесорів вхідних маршрутів. Якщо збережений граф містить фрагмент моделі, Neat зберігає шлях до архіву моделі й параметри маршруту, потрібні, щоб відновити цей фрагмент. Якщо під час завантаження архів відсутній, Neat завершується помилкою з підказкою, що робити, а не будує непомітно неповний маршрут.

## Експорт і візуалізація виконаного

`Run` знає як публічну структуру графа, так і понижену (lowered) структуру середовища виконання. Його можна експортувати як версійований артефакт JSON для CI, налагодження, звернень до служби підтримки або офлайн-візуалізації.

### Знімок топології під час побудови

Використовуйте експорт під час побудови, якщо артефакт потрібен одразу після побудови графа:

```cpp
neat::RunOptions opt;
opt.run_export.path = "/tmp/startup.graph_run.json";
opt.run_export.label = "startup";

neat::Run run = app.build(opt);
```

Це початковий знімок топології. Лічильники пропускної здатності й затримки в ньому можуть бути нульовими, бо через граф ще не пройшов жоден зразок.

Python:

```python
opt = pyneat.RunOptions()
opt.run_export.path = "/tmp/startup.graph_run.json"
opt.run_export.label = "startup"

run = app.build(opt)
```

### Знімок із метриками після виконання

Використовуйте експорт після виконання, коли навантаження вже виконано або виходи спорожнено:

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

Експортер робить знімок поточного запуску, але не зупиняє його. Якщо для скінченного навантаження потрібні остаточні цифри, перед збереженням викличте `run.close_input()` і спорожніть виходи або викличте `run.stop()`.

Застарілий `RunOptions::enable_board_power()` додає до експорту споживану потужність плати. У новому коді записуйте споживану потужність плати, температури, завантаження CPU та пам’ять за допомогою [`TelemetryMonitor`](/develop-apps/development-workflow/model#record-board-telemetry) і зберігайте його `Telemetry` поруч з експортом.

Схема JSON — `sima.neat.graph_run` версії `1`. Схема розташована в `schemas/graph_run_v1.schema.json`, а валідатор CI — у `tests/perf/tools/graph_run_schema.py`.

Відобразіть артефакт без доступу до інтернету:

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json -o /tmp/final.graph_run.html
```

Виберіть подання для відображення:

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view public
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view lowered
```

- `public` показує граф, створений користувачем: іменовані входи, виходи, фрагменти та ребра
  `connect(...)`.
- `lowered` показує, що Neat виконав усередині: сегменти конвеєра, згенеровані етапи розгалуження/об’єднання,
  черги та ребра середовища виконання.

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

### Від камери RTSP через модель до виходу, з якого дані отримує застосунок

```cpp
neat::nodes::groups::RtspDecodedInputOptions source_opt;
source_opt.url = "rtsp://camera/stream";

neat::Graph app;
app.add(neat::nodes::groups::RtspDecodedInput(source_opt));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

Група декодованого RTSP-входу надає однакові параметри керування декодером у C++ і Python:

| Поле `RtspDecodedInputOptions` | За замовчуванням | Призначення |
| --- | --- | --- |
| `decoder_input_buffers` | `-1` | Кількість буферів стисненого входу; якщо не задано, діє значення середовища виконання за замовчуванням. |
| `num_buffers` | `-1` | Кількість буферів у пулі декодованого виходу; якщо не задано, розмір визначається автоматично. |
| `decoder_tuning` | `""` | Необов’язковий профіль декодера, наприклад `throughput-low-latency`. |
| `decoder_memory_opt` | `false` | Застарілий профіль пам’яті; явно заданий `decoder_tuning` має пріоритет. |

Додатні перевизначення кількості діють і з профілем налаштування. Використовуйте `-1`, щоб не задавати кількість; інші недодатні значення, які використовувалися раніше, також залишають властивість незаданою.
Профіль `throughput-low-latency` вимикає перевпорядкування й потребує сумісних потоків без перевпорядкування B-кадрів для показу. Кількість вхідних буферів не задає їхню місткість у байтах.
Ці параметри вже є в `SimaDecodeOptions` як `input_buffers`,
`num_buffers`, `decoder_tuning` та `memory_opt`.

```python
source = pyneat.RtspDecodedInputOptions()
source.url = "rtsp://camera/stream"
source.decoder_input_buffers = 2
source.decoder_tuning = "throughput-low-latency"
# Leave num_buffers at -1 to use automatic output sizing.
app.add(pyneat.groups.rtsp_decoded_input(source))
```

### Від входу застосунку до виходу UDP, яким володіє граф

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::UdpOutput(udp_options));
```

### Розгалуження на попередній перегляд і шлях моделі

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_image"});
```

Назвіть `model_image` відповідно до входу маршруту моделі або вставте явний фрагмент-адаптер.

### Об’єднання лівого й правого потоків

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "pair",
                                         neat::CombinePolicy::ByPts);
```

Використовуйте `ByPts`, коли ключем синхронізації є медіамітки часу, і `ByFrame`, коли ключем
синхронізації є ідентифікатори кадрів.

### GenAI та інші фрагменти з етапами

GenAI та інші нелінійні або поетапні можливості однаково мають потрапляти в код застосунку як публічні
фрагменти `Graph` і виконуватися через `Graph::build() -> Run`:

```cpp
neat::Graph app;
app.add(genai_fragment);

neat::Run run = app.build();
run.push("prompt", prompt_sample);
auto token = run.pull("tokens", 1000);
```

Точні назви фабрики фрагментів GenAI і допоміжних функцій для зразків залежать від установленого пакета GenAI. Правило Graph те саме: додайте або з’єднайте публічні фрагменти, а потім використовуйте іменовані `Run::push(...)` і `Run::pull(...)`.

## Антипатерни та підводні камені

### Не використовуйте мітки графа як кінцеві точки

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

### Не вгадуйте назви входів моделі

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

Потім дайте кінцевим точкам вище за потоком відповідні назви.

### Не використовуйте push/pull без назви в графах із кількома кінцевими точками

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

### Не зводьте потоки ненавмисно без CombinePolicy

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

### Не вставляйте Input/Output посередині, якщо не маєте на увазі межу фрагмента

`Input` і `Output` — це публічні оголошення меж. У повторно використовуваних фрагментах саме це й потрібно. У суто лінійному застосунку додатковий `Output` посередині може створити справжній приймач, з якого можна отримувати дані (pull), і зворотний тиск, якщо цю межу не споживає інше ребро `connect(...)`.

### Не використовуйте в коді застосунку низькорівневі API графа середовища виконання

Не пояснюйте й не пишіть код застосунків із використанням низькорівневих API графа середовища виконання.

Натомість використовуйте публічний інтерфейс застосунку:

```cpp
neat::Graph
neat::Run
app.build()
```

## Поглиблена примітка: матеріалізація меж

Іменовані вузли `Input` і `Output` — це оголошення публічного контракту фрагмента. Вони розташовані на вищому рівні, ніж об’єкти середовища виконання, які переміщують буфери.

Перед побудовою виконуваного конвеєра `Graph::build()` нормалізує межі:

| Оголошення межі | Матеріалізується, коли... | Опускається, коли... |
|---|---|---|
| `nodes::Input("name")` | до нього не під’єднано жодного графа вище за потоком, тому це має бути публічна кінцева точка `Run::push("name", ...)` | його живить граф вище за потоком, тому це лише внутрішній параметр фрагмента |
| `nodes::Output("name")` | жоден граф нижче за потоком його не споживає, тому це має бути публічна кінцева точка `Run::pull("name")` | його споживає граф нижче за потоком, тому це лише внутрішнє значення, яке повертає фрагмент |

Опущено не означає забуто. Компілятор зберігає походження, тому `describe()`, помилки перевірки, метрики та JSON експорту запуску й надалі можуть посилатися на назву кінцевої точки, яку бачить користувач.

Це не дає повторно використовуваним фрагментам створювати посеред застосунку приховане введення/виведення середовища виконання на кшталт appsrc/appsink. Наприклад:

```cpp
neat::Graph app;
app.connect(camera, route);
app.connect(route, display);
```

Виконуваний шлях даних — `camera -> route body -> display`, а не `camera -> route.Input -> route.Output -> display`
з додатковими фізичними приймачами/джерелами посередині.

## Короткий довідник API

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

## Додаткові матеріали

- [Модель](/develop-apps/development-workflow/model)
- [Вузол: групи й межі](/develop-apps/development-workflow/node#boundary-nodes)
- [Тензор і зразок](/develop-apps/development-workflow/core_types)
- [Налаштування середовища виконання (посібник 016)](/tutorials/tune-throughput-and-queues)
- [Діагностика (посібник 012)](/tutorials/diagnose-a-pipeline)
- [Рівень GStreamer](/develop-apps/advanced-concepts/gstreamer_layer)
