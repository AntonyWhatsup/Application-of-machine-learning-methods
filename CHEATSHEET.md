# GitHub Commands Cheat Sheet / Шпаргалка по командах GitHub

This document contains a list of essential Git and GitHub commands with explanations and examples.

---

## English Version

### 1. Starting a Repository
| Command | Explanation | Example |
| :--- | :--- | :--- |
| `git init` | Initializes a new local Git repository in the current directory. | `git init` |
| `git clone [url]` | Downloads a project and its entire version history from a remote URL. | `git clone https://github.com/user/repo.git` |

### 2. Making Changes
| Command | Explanation | Example |
| :--- | :--- | :--- |
| `git status` | Shows the status of changes as untracked, modified, or staged. | `git status` |
| `git add [file]` | Adds a specific file to the staging area. | `git add index.html` |
| `git add .` | Adds all modified and new files to the staging area. | `git add .` |
| `git commit -m "[message]"` | Records the staged snapshots in the version history with a descriptive message. | `git commit -m "Add navigation bar"` |

### 3. Branches & Merging
| Command | Explanation | Example |
| :--- | :--- | :--- |
| `git branch` | Lists all local branches in the current repository. | `git branch` |
| `git branch [branch-name]` | Creates a new branch. | `git branch feature-login` |
| `git checkout [branch-name]` | Switches to the specified branch. | `git checkout feature-login` |
| `git checkout -b [branch-name]` | Creates a new branch and switches to it immediately. | `git checkout -b fix-bug` |
| `git merge [branch]` | Combines the specified branch's history into the current branch. | `git merge feature-login` |

### 4. Remote Repositories
| Command | Explanation | Example |
| :--- | :--- | :--- |
| `git remote add origin [url]` | Connects your local repository to a remote server. | `git remote add origin https://github.com/user/repo.git` |
| `git push -u origin [branch]` | Sends local branch commits to the remote repository. | `git push -u origin main` |
| `git pull` | Fetches and merges changes from the remote server to your local directory. | `git pull` |

### 5. Reviewing History
| Command | Explanation | Example |
| :--- | :--- | :--- |
| `git log` | Lists the version history for the current branch. | `git log` |
| `git diff` | Shows content differences between the working directory and the last commit. | `git diff` |

---

## Українська версія

### 1. Початок роботи з репозиторієм
| Команда | Пояснення | Приклад |
| :--- | :--- | :--- |
| `git init` | Ініціалізує новий локальний репозиторій Git у поточному каталозі. | `git init` |
| `git clone [url]` | Завантажує проєкт та всю історію його версій з віддаленої URL-адреси. | `git clone https://github.com/user/repo.git` |

### 2. Внесення змін
| Команда | Пояснення | Приклад |
| :--- | :--- | :--- |
| `git status` | Показує стан змін: невідстежувані, змінені або додані до індексу. | `git status` |
| `git add [файл]` | Додає конкретний файл до області підготовки (staging area). | `git add index.html` |
| `git add .` | Додає всі змінені та нові файли до області підготовки. | `git add .` |
| `git commit -m "[повідомлення]"` | Записує знімки підготовлених змін в історію версій із описовим повідомленням. | `git commit -m "Додано панель навігації"` |

### 3. Гілки та злиття
| Команда | Пояснення | Приклад |
| :--- | :--- | :--- |
| `git branch` | Виводить список усіх локальних гілок у поточному репозиторії. | `git branch` |
| `git branch [назва-гілки]` | Створює нову гілку. | `git branch feature-login` |
| `git checkout [назва-гілки]` | Перемикається на вказану гілку. | `git checkout feature-login` |
| `git checkout -b [назва-гілки]` | Створює нову гілку і відразу перемикається на неї. | `git checkout -b fix-bug` |
| `git merge [гілка]` | Об'єднує історію вказаної гілки з поточною гілкою. | `git merge feature-login` |

### 4. Віддалені репозиторії
| Команда | Пояснення | Приклад |
| :--- | :--- | :--- |
| `git remote add origin [url]` | Підключає ваш локальний репозиторій до віддаленого сервера. | `git remote add origin https://github.com/user/repo.git` |
| `git push -u origin [гілка]` | Відправляє коміти локальної гілки до віддаленого репозиторію. | `git push -u origin main` |
| `git pull` | Отримує та об'єднує зміни з віддаленого сервера у ваш локальний каталог. | `git pull` |

### 5. Перегляд історії
| Команда | Пояснення | Приклад |
| :--- | :--- | :--- |
| `git log` | Виводить список історії версій для поточної гілки. | `git log` |
| `git diff` | Показує різницю у вмісті між робочим каталогом і останнім комітом. | `git diff` |