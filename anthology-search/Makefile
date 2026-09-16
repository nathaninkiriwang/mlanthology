.DEFAULT_GOAL := help
UV ?= uv

.PHONY: index reindex refresh lanes doctor test venues stats mcp help

index:  ## Build the FTS5 index from ../data (skips if already current)
	$(UV) run mla index

reindex:  ## Force a rebuild even if the corpus has not moved
	$(UV) run mla index --force

refresh:  ## Pull this fork's latest corpus data, then rebuild the index
	# origin, not upstream: upstream lacks the ACL adapter and this fork has diverged
	# from it, so --ff-only against upstream can never succeed. The scheduled
	# com.local.mlarefresh agent does this automatically; this stays for manual use.
	git -C .. fetch origin && git -C .. merge --ff-only origin/main && $(UV) run mla index

lanes:  ## Rebuild the BM25 lane and top up the default dense lane (hybrid extra)
	$(UV) run mla bm25 build >/dev/null && $(UV) run mla vec status | head -40
	@d=$$(cat index/vec/DEFAULT 2>/dev/null); [ -n "$$d" ] && $(UV) run mla vec build --model $$d --quiet >/dev/null || echo "no default dense lane (mla vec default <model>)"

doctor:  ## Is the index present, fresh, and answering?
	$(UV) run mla doctor

venues:  ## Venue coverage table
	$(UV) run mla venues --text

stats:  ## Index provenance and field coverage
	$(UV) run mla stats

mcp:  ## Run the secondary MCP server on stdio
	$(UV) run --extra mcp mla-mcp

test:  ## Hermetic tests against a synthetic corpus (no index needed)
	$(UV) run --with pytest pytest tests -q

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
