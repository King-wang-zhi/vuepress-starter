// node_modules/.pnpm/vuepress-theme-reco@2.0.0-rc.26_@algolia+client-search@5.28.0_@types+node@24.0.3_@vueus_7b77752834b28d7a7269686f3a1a3de2/node_modules/vuepress-theme-reco/lib/client/utils/other.js
function formatISODate(ISODate = "") {
  const dateStr = ISODate.replace("T", " ").replace("Z", "").split(".")[0];
  const formatDateStr = dateStr.replace(/(\s00:00:00)$/, "");
  return formatDateStr;
}

export {
  formatISODate
};
//# sourceMappingURL=chunk-QJUFEH3V.js.map
