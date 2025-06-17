
  import { defineAsyncComponent } from 'vue'
import { defineClientConfig } from 'vuepress/client'

import { applyClientSetup } from '/Users/sunluyu/Documents/study/vuepress-starter/node_modules/.pnpm/vuepress-theme-reco@2.0.0-rc.26_@algolia+client-search@5.28.0_@types+node@24.0.3_@vueus_7b77752834b28d7a7269686f3a1a3de2/node_modules/vuepress-theme-reco/lib/client/clientSetup.js'
import { applyClientEnhance } from '/Users/sunluyu/Documents/study/vuepress-starter/node_modules/.pnpm/vuepress-theme-reco@2.0.0-rc.26_@algolia+client-search@5.28.0_@types+node@24.0.3_@vueus_7b77752834b28d7a7269686f3a1a3de2/node_modules/vuepress-theme-reco/lib/client/clientEnhance.js'

import * as layouts from '/Users/sunluyu/Documents/study/vuepress-starter/node_modules/.pnpm/vuepress-theme-reco@2.0.0-rc.26_@algolia+client-search@5.28.0_@types+node@24.0.3_@vueus_7b77752834b28d7a7269686f3a1a3de2/node_modules/vuepress-theme-reco/lib/client/layouts/index.js'

  const layoutsFromDir = {}
export default defineClientConfig({
  enhance(...args) {
    applyClientEnhance(...args)
  },
  setup() {
    applyClientSetup()
  },
  // @ts-ignore
  layouts: { ...layouts, ...layoutsFromDir },
})
