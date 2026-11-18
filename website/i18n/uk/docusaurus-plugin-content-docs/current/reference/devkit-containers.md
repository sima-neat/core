---
title: "Контейнери DevKit"
description: "Встановлення Docker на Modalix DevKit, команди dk container, контейнери Ubuntu та усунення несправностей"
sidebar_position: 11
---

# Контейнери DevKit

Ця сторінка є довідником щодо запуску контейнерів на Modalix DevKit із
версією платформи 3.0. Огляд і повний робочий процес наведено в розділі
[Розробка контейнеризованих застосунків](/getting-started/dev-environment/containerized-apps/).

## Доступ до прискорювача {#accelerators}

Застосунки Neat можуть використовувати MLA в контейнері. Контейнер повинен мати:

- Доступ до потрібних пристроїв, наприклад `/dev/mla`, `/dev/cvu` і `/dev/dma_heap/linux,cma`, переданих за допомогою параметра Docker `--device`.
- Сумісні бібліотеки Neat, плагіни GStreamer та бібліотеки користувацького простору платформи, упаковані в образ або змонтовані з DevKit.
- Сумісну платформу DevKit і версію Neat Core, яка підтримує визначення платформи в контейнері. Застосунок має відповідати ABI середовища виконання.

На оновленому DevKit запускайте навантаження з UID і GID користувача DevKit за допомогою `--user UID:GID` та з дозволом доступу до переданих пристроїв. Після налаштування доступу до Docker застосунку не потрібні `sudo` або `--privileged` для доступу до MLA. Поточний офіційний засіб запуску також не використовує параметрів, що знімають обмеження безпеки. Доступ адміністратора для встановлення Docker й налаштування хоста відокремлений від дозволів застосунку на пристрої.

Драйвери ядра та мікропрограму й надалі надає DevKit. Саме лише передавання пристроїв не робить довільний образ Ubuntu готовим до використання прискорювача. Навантаження DSP потребують власного сумісного середовища виконання та налаштування пристроїв; цей посібник не встановлює підтримку DSP для довільних образів контейнерів.

Повну конфігурацію пристроїв і бібліотек наведено в офіційному [прикладі паралельних застосунків у Docker](https://github.com/sima-neat/apps/tree/develop/examples/docker/parallel-apps). Дотримуйтеся передумов для вибраного прикладу та версії платформи.

## Встановлення Docker на DevKit {#install}

Встановіть Docker із Palette SDK або за допомогою автономного інсталятора. Нове встановлення будь-яким із наведених нижче способів:

- Встановлюють Docker CE (`docker-ce`, `docker-ce-cli` і `containerd.io`) для
  ARM64 з Debian-репозиторію Docker `trixie`.
- Зберігають дані Docker у `/data/docker`, а дані containerd — у
  `/data/containerd`, щоб образи не заповнили кореневу файлову систему.
- Вмикають переадресацію IP, потрібну для мережевої взаємодії контейнерів.
- Запускають служби `docker` і `containerd` та налаштовують їх запуск під час кожного завантаження.
- Додають користувача DevKit до групи `docker`, щоб користувач міг виконувати `docker`
  без `sudo`.

Обидва способи потребують доступу до Інтернету на DevKit.

### З Palette SDK {#install-sdk}

Спочатку створіть пару DevKit за допомогою `sima-cli sdk setup --devkit {devkit-ip}`. Це налаштовує доступ до `sudo` без пароля, потрібний для встановлення. Потім в оболонці SDK виконайте:

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

Дайте відповідь `y` на пропозицію налаштувати Docker на DevKit. Щоб пропустити запитання, виконайте `dk container setup --yes`. Якщо Docker уже працює, команда використає його повторно. Якщо встановлення перервано, повторіть команду, щоб продовжити.

Інші команди `dk container` також пропонують встановити Docker, якщо його немає.

### Ручне встановлення на DevKit {#install-manual}

Якщо Palette SDK недоступний, скористайтеся автономним інсталятором SDK. На ARM64 Modalix DevKit, потрібні доступ до Інтернету й обліковий запис із доступом до sudo. Через SSH або послідовну консоль завантажте інсталятор із репозиторію SDK та запустіть його як звичайний користувач:

<ShellCommand prompt="devkit">
curl --fail --location --output install-devkit-docker.sh \
  https://raw.githubusercontent.com/sima-neat/sdk/main/scripts/install-devkit-docker.sh
bash install-devkit-docker.sh install
</ShellCommand>

Автономний режим `install` запитує пароль sudo за потреби. Він використовує те саме налаштування Docker і відновлення після переривання, що й `dk container setup`, зокрема конфігурацію сховища та відновлення дзеркала пакетів eLxr. Наявна конфігурація Docker зберігається. Якщо встановлення перервано, повторіть ту саму команду.

Вийдіть із системи й увійдіть знову, щоб активувати членство в групі Docker, а потім [перевірте встановлення](#check).

## Команди dk container {#dk-container}

Виконуйте ці команди в оболонці Palette SDK. Вони підключаються до DevKit, з яким створено пару,
через SSH і запускають там Docker.

| Команда | Що вона робить | Потребує локального реєстру |
| --- | --- | --- |
| `dk container setup [--yes]` | Встановлює Docker на DevKit або залишає робоче встановлення без змін. | Ні |
| `dk container deploy <image> [options] [-- command]` | Завантажує образ із локального реєстру на DevKit, а потім запускає його. | Так |
| `dk container run <image> [options] [-- command]` | Запускає образ, який уже є на DevKit. | Так |
| `dk container pull <image>` | Завантажує образ із локального реєстру на DevKit. | Так |
| `dk container images` | Показує образи на DevKit, отримані з локального реєстру. | Так |
| `dk container list` | Показує всі контейнери на DevKit, запущені та зупинені. | Ні |
| `dk container logs <container> [options]` | Показує вивід контейнера. Додайте `--follow`, щоб продовжувати стежити за ним. | Ні |
| `dk container stop <container>` | Зупиняє контейнер. | Ні |
| `dk container remove <container> [options]` | Видаляє контейнер. Також працює `rm`. | Ні |
| `dk container help` | Показує стислий опис команд. | Ні |

Як команди обробляють свої аргументи:

- **Назви образів.** Вказуйте назву образу без адреси реєстру, наприклад
  `my-app:latest`. `dk` додає адресу локального реєстру, яку використовує DevKit.
  Тому `deploy`, `run` і `pull` працюють лише з образами з локального реєстру.
- **Параметри.** Аргументи після назви образу й до `--` є параметрами `docker run`,
  як-от `--detach`, `--name`, `--rm` або `--network host`. Аргументи
  після `--` — це команда, яку потрібно виконати в контейнері.
- **Архітектура.** `deploy` і `run` відхиляють образ, який не є образом ARM64.

Наприклад, щоб запустити образ і перевизначити його команду:

<ShellCommand prompt="sdk">
dk container run my-app:latest --rm -- sh -c 'uname -m'
</ShellCommand>

## Запуск контейнера Ubuntu {#ubuntu}

Образи Ubuntu з Docker Hub працюють на Arm-процесорах DevKit. Щоб спробувати один із них, виконайте цю
команду на DevKit:

<ShellCommand prompt="devkit">
docker run --rm ubuntu:24.04 cat /etc/os-release
</ShellCommand>

Вона виведе `PRETTY_NAME="Ubuntu 24.04.5 LTS"` або новіший проміжний випуск. Для
інтерактивної оболонки виконайте `docker run --rm -it ubuntu:24.04 bash`. Усередині
контейнера `apt-get` встановлює пакети Ubuntu для ARM64.

Контейнер використовує ядро Linux DevKit; це не окреме встановлення операційної
системи. Файли, змінені в контейнері, буде втрачено, коли ви його видалите.
Щоб зберегти файли, змонтуйте каталог DevKit у контейнер за допомогою `-v`.

Щоб постачати застосунок на Ubuntu, почніть його файл `Dockerfile` з
`FROM ubuntu:<release>`, а потім зберіть і розгорніть його з SDK, як показано в розділі
[Розробка контейнеризованих застосунків](/getting-started/dev-environment/containerized-apps/).
Команда `dk container run ubuntu:24.04` не працює, тому що `dk` шукає образ
у локальному реєстрі.

Наведені вище команди Ubuntu демонструють лише виконання на CPU. Для навантажень Neat із прискорювачем дотримуйтеся розділу [Доступ до прискорювача](#accelerators); сама лише зміна базового образу не забезпечує потрібних пристроїв і бібліотек середовища виконання.

## Перевірка встановлення {#check}

На DevKit:

<ShellCommand prompt="devkit">
docker info --format '{{.DockerRootDir}}'
docker run --rm hello-world
</ShellCommand>

Перша команда виведе `/data/docker`. Друга виведе `Hello from Docker!`.

## Вільне місце в /data {#storage}

Docker зберігає образи та контейнери в `/data`, розмір якого становить близько 3 ГБ. Перевірте
вільне місце на DevKit:

<ShellCommand prompt="devkit">
df -h /data
</ShellCommand>

Щоб звільнити місце, видаліть контейнери та образи, які вам більше не потрібні:

<ShellCommand prompt="devkit">
docker container prune
docker image prune -a
</ShellCommand>

Команда `docker image prune -a` видаляє всі образи, які не використовує жоден контейнер.

## Усунення несправностей {#troubleshooting}

| Повідомлення | Виправлення |
| --- | --- |
| `The local container registry is not configured for this SDK shell.` | Виконайте на хості `sima-cli sdk setup --devkit {devkit-ip}`, дайте відповідь `y` на запитання про реєстр контейнерів, а потім відкрийте нову оболонку SDK. |
| `Docker installation requires passwordless sudo` | Повторно створіть пару за допомогою `sima-cli sdk setup --devkit {devkit-ip}` або скористайтеся [автономним інсталятором](#install-manual). |
| `Image ... is for amd64, not ARM64.` | Перезберіть образ за допомогою `docker buildx build --platform linux/arm64 ... --push`. |
| `Image ... is not available on the DevKit.` | Спочатку завантажте його за допомогою `dk container deploy` або `dk container pull`. |
| `permission denied while trying to connect to the Docker daemon socket` на DevKit | Після налаштування Docker вийдіть із системи й увійдіть знову, щоб ваш користувач опинився в групі `docker`. |
| `exec format error` під час збирання | Кроку `RUN` у файлі `Dockerfile` потрібна емуляція ARM64 на хості `x86_64`. Встановіть її на хості за допомогою `docker run --privileged --rm tonistiigi/binfmt --install arm64`, а потім повторіть збирання. |
