# 003 Виміряйте продуктивність моделі

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5 хвилин |
| Model | resnet_50 |
| Labels | benchmark, synthetic, latency, throughput, power |

## Concept

Виміряйте затримку, пропускну здатність, споживану потужність та енергію скомпільованої моделі за допомогою `model.benchmark()`. Тест створює синтетичні вхідні дані, тому зображення чи набір даних не потрібні.

## Walkthrough

### Завантажте модель {#step-load-model}

Завантажте скомпільований архів `.tar.gz` через `simaai::neat::Model` у C++ або `pyneat.Model` у Python. Модель має оголошувати конкретні розміри вхідних даних у `input_specs()`.

### Запустіть тест продуктивності {#step-run-benchmark}

Викличте `model.benchmark()` для типових налаштувань або `model.benchmark(100)`, щоб вибрати кількість зразків. API прогріває модель, вимірює затримку послідовного виконання та пропускну здатність асинхронного виконання, виводить підсумок і повертає `BenchmarkReport`.

Більша кількість зразків може зробити вимірювання пропускної здатності та потужності стабільнішими.

### Перегляньте звіт {#step-read-report}

| Поле | Значення | Одиниця |
| --- | --- | --- |
| `latency_ms` | Середня затримка після прогрівання | ms |
| `fps` | Пропускна здатність логічних інференсів | inferences/s |
| `avg_power_watts` | Середня споживана потужність плати під час вимірювання пропускної здатності | W |
| `energy_joules` | Загальна спожита енергія під час вимірювання пропускної здатності | J |

Якщо телеметрія споживаної потужності плати недоступна, значення потужності та енергії залишаються нульовими.

У коді Python ви також можете зберегти звіт:

```python
model = pyneat.Model("model.tar.gz")
report = model.benchmark()
report.save_json("benchmark.json")
```

Використовуйте `report.to_json()`, щоб отримати текст JSON, або `print(report)`, щоб знову показати звіт. JSON явно позначає недоступні вимірювання, а не подає їх як дійсні нульові значення.

## Run

Запускайте приклади Python і попередньо зібраний C++ із кореневої теки встановлення Neat, яка містить `share/` і `lib/`. Команди збірки з вихідного коду запускайте з кореневої теки репозиторію. Обидва приклади виводять підсумок тесту, а потім поля повернутого звіту.

**Python:**
```bash
python3 share/sima-neat/tutorials/003_benchmark_your_model/benchmark_your_model.py \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_003_benchmark_your_model
./build/tutorials-standalone/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

Приклад полів звіту з умовними числами:

```text
report_latency_ms=12.4
report_fps=80.6
report_avg_power_watts=2.3
report_energy_joules=2.8
```

Для власного проєкту C++ див. [Як запускати навчальні приклади](/tutorials#compile-a-copy-yourself).

## In Practice

Порівнюйте результати з однаковими налаштуваннями моделі та на одному пристрої. Записуйте архів моделі й кількість зразків разом зі збереженими звітами.

Синтетичні вхідні дані не відтворюють часові характеристики камери чи обробку, залежну від даних. Окремо вимірюйте реальний конвеєр застосунку. Для налаштування асинхронної черги див. [Налаштування пропускної здатності та глибини черги](/tutorials/tune-throughput-and-queues).

Про розміри початкового зображення для BoxDecode та інші параметри тесту див. [Вимірювання продуктивності моделі](/develop-apps/development-workflow/model#measure-model-execution).

## Файли вихідного коду
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
