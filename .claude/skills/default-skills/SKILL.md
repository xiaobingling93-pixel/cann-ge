---
name: default-skills
description: |
  管理项目必备技能的安装和维护。

  使用场景：
  - 新环境首次配置
  - 技能更新后重新安装
  - 修改技能列表

  **重要提示**：自动执行 install-default-skills.sh 安装项目必需技能。
---

## 功能

### 安装技能
自动安装默认技能：

### 修改技能列表
编辑 `scripts/install-default-skills.sh` 中的 `DEFAULT_SKILLS` 数组：

**添加技能**：
```bash
DEFAULT_SKILLS=("gitcode-pr" "gitcode-issue" "new-skill")
```

**移除技能**：
```bash
DEFAULT_SKILLS=("gitcode-pr")
```

## 使用

直接使用 `/skill default-skills` 安装或重新安装技能。

## 故障排除

- **npx 未找到**：安装 Node.js
- **访问失败**：检查网络和仓库 URL
- **权限错误**：运行 `chmod +x scripts/install-default-skills.sh`