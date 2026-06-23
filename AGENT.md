# AGENT 协作约定

本文件记录与 AI 助手（Claude Code 等）协作时的固定约定，每次会话生效。

## 提交规范

- **每完成一项操作即提交代码**：一个完整、可独立描述的改动完成后立即 `git commit`，不要把多个无关改动堆在一起。
- **提交信息不要把 Claude Code 列为 Co-Author**：禁止在 commit message 末尾添加 `Co-Authored-By: Claude ...` 之类的署名行。
