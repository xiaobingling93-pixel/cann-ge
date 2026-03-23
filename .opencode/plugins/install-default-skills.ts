import type { Plugin } from "@opencode-ai/plugin"
import * as fs from "fs"
import * as path from "path"

const logFile = path.join(__dirname, "log.txt")

function log(message: string) {
  // const timestamp = new Date().toISOString()
  // fs.appendFileSync(logFile, `[${timestamp}] ${message}\n`)
}

export const InstallSkillsPlugin: Plugin = async ({ $, project }) => {
  log("InstallSkillsPlugin loaded")
  return {
    event: async ({ event }) => {
      if (event.type === "session.created") {
        log("Executing install-default-skills.sh")
        await $`bash ../../.claude/skills/default-skills/scripts/install-default-skills.sh > /dev/null 2>&1`
      }
    }
  }
}
