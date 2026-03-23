#!/bin/bash

# 脚本功能：安装项目必备的 skills
# 使用方法：直接运行此脚本

echo "正在安装项目必备的 skills..."

# 默认技能列表
DEFAULT_SKILLS=("gitcode-pr" "gitcode-issue")

echo "将要安装的技能: ${DEFAULT_SKILLS[*]}"

# 执行安装命令
npx skills add https://gitcode.com/cann-agent/skills.git -y -g -s ${DEFAULT_SKILLS[*]}