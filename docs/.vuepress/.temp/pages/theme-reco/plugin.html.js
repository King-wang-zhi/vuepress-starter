import comp from "/Users/sunluyu/Documents/study/vuepress-starter/docs/.vuepress/.temp/pages/theme-reco/plugin.html.vue"
const data = JSON.parse("{\"path\":\"/theme-reco/plugin.html\",\"title\":\"plugin\",\"lang\":\"zh-CN\",\"frontmatter\":{\"title\":\"plugin\",\"date\":\"2020/05/28\"},\"git\":{},\"filePathRelative\":\"theme-reco/plugin.md\",\"excerpt\":\"<p>This is plugin.</p>\\n\"}")
export { comp, data }

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept()
  if (__VUE_HMR_RUNTIME__.updatePageData) {
    __VUE_HMR_RUNTIME__.updatePageData(data)
  }
}

if (import.meta.hot) {
  import.meta.hot.accept(({ data }) => {
    __VUE_HMR_RUNTIME__.updatePageData(data)
  })
}
