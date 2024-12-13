 int dissolve_free_huge_pages(unsigned long start_pfn, unsigned long end_pfn)
 {
 	unsigned long pfn;
	struct page *page;
 	int rc = 0;
 
 	if (!hugepages_supported())
 		return rc;
 
	for (pfn = start_pfn; pfn < end_pfn; pfn += 1 << minimum_order) {
		page = pfn_to_page(pfn);
		if (PageHuge(page) && !page_count(page)) {
			rc = dissolve_free_huge_page(page);
			if (rc)
				break;
		}
	}
 
 	return rc;
 