import comp from "/Users/sunluyu/Documents/study/vuepress-starter/docs/.vuepress/.temp/pages/posts.html.vue"
const data = JSON.parse("{\"path\":\"/posts.html\",\"title\":\"\",\"lang\":\"zh-CN\",\"frontmatter\":{\"layout\":\"Posts\"},\"git\":{},\"filePathRelative\":null,\"excerpt\":\"\"}")
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
