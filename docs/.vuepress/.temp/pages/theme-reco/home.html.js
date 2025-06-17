import comp from "/Users/sunluyu/Documents/study/vuepress-starter/docs/.vuepress/.temp/pages/theme-reco/home.html.vue"
const data = JSON.parse("{\"path\":\"/theme-reco/home.html\",\"title\":\"theme-reco\",\"lang\":\"zh-CN\",\"frontmatter\":{\"title\":\"theme-reco\",\"date\":\"2020/05/29\"},\"git\":{},\"filePathRelative\":\"theme-reco/home.md\",\"excerpt\":\"<p>This is theme-reco.</p>\\n\"}")
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
