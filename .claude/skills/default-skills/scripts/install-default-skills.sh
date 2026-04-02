#!/bin/bash

# 脚本功能：安装项目必备的 skills
# 默认技能列表
DEFAULT_SKILLS=("gitcode-pr" "gitcode-issue")

# 执行安装命令（安装到 OpenCode）
npx skills add https://gitcode.com/cann-agent/skills.git -y -a claude-code -s ${DEFAULT_SKILLS[*]}