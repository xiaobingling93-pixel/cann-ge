import type { Plugin } from "@opencode-ai/plugin"

export const InstallSkillsPlugin: Plugin = async ({ $, project }) => {
  return {
    event: async ({ event }) => {
      if (event.type === "session.created") {
        await $`bash ./.claude/skills/default-skills/scripts/install-default-skills.sh > /dev/null`
      }
    }
  }
}
