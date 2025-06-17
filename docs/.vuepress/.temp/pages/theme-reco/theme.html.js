import comp from "/Users/sunluyu/Documents/study/vuepress-starter/docs/.vuepress/.temp/pages/theme-reco/theme.html.vue"
const data = JSON.parse("{\"path\":\"/theme-reco/theme.html\",\"title\":\"theme\",\"lang\":\"zh-CN\",\"frontmatter\":{\"title\":\"theme\",\"date\":\"2020/05/27\"},\"git\":{},\"filePathRelative\":\"theme-reco/theme.md\",\"excerpt\":\"<p>This is theme.</p>\\n\"}")
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
