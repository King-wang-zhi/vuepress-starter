import comp from "/Users/sunluyu/Documents/study/vuepress-starter/docs/.vuepress/.temp/pages/theme-reco/api.html.vue"
const data = JSON.parse("{\"path\":\"/theme-reco/api.html\",\"title\":\"api\",\"lang\":\"zh-CN\",\"frontmatter\":{\"title\":\"api\",\"date\":\"2020/05/29\"},\"git\":{},\"filePathRelative\":\"theme-reco/api.md\",\"excerpt\":\"<p>This is api.</p>\\n\"}")
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
