#!/bin/bash
# Fetch the interaction networks on the cluster rather than through the laptop.
#
# HIPPIE and BioGRID are 139 MB and ~200 MB; there is no reason for either to
# cross a home connection when the analysis runs here.  HuRI is deliberately
# absent: interactome-atlas.org redirects to an https endpoint whose certificate
# does not match its host, and the fix for that is not --insecure.  If HuRI is
# wanted, take it from a mirror with a valid certificate (IntAct/IMEx) or ask
# the authors, and add it here once resolved.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p data && cd data || exit 1

get () {   # get <url> <outfile>
    if [ -s "$2" ]; then echo "have $2"; return; fi
    echo "fetching $2"
    curl -sSL --max-time 1800 -o "$2.part" "$1" && mv "$2.part" "$2" \
        || { echo "FAILED $2"; rm -f "$2.part"; }
}

# STRING physical subnetwork, human.  In UPNA-PPI's own source list, and the
# only one small enough that it was already fetched locally.
get "https://stringdb-downloads.org/download/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz" \
    string_physical_9606.txt.gz

# HIPPIE: literature-curated, dense, and carrying the study bias that STRING
# physical at high confidence partly suppresses.  The contrast is the point --
# it separates what the sampling rule does from what the network already did.
get "https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/hippie_current.txt" \
    hippie_current.txt

# BioGRID human, physical interactions only, from the current release archive.
get "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.246/BIOGRID-ORGANISM-4.4.246.tab3.zip" \
    biogrid_tab3.zip

# ---- normalise everything to two-column edge lists -------------------------

if [ -s string_physical_9606.txt.gz ]; then
    for t in 700 900; do
        [ -s "string_phys_${t}.tsv" ] && continue
        gunzip -c string_physical_9606.txt.gz \
            | awk -v t=$t 'NR>1 && $3>=t {print $1"\t"$2}' > "string_phys_${t}.tsv"
        echo "string_phys_${t}.tsv  $(wc -l < string_phys_${t}.tsv) rows"
    done
fi

# HIPPIE's own tab format, no header:
#   1 Symbol A   2 Entrez A   3 Symbol B   4 Entrez B   5 score   6 sources
# The two ENDPOINTS are columns 1 and 3.  Columns 1 and 2 are the same protein
# written two ways -- pairing those produced a graph of 34,095 nodes with max
# degree 14, which is what the sanity check below now catches.  0.63 is HIPPIE's
# own high-confidence mark.
if [ -s hippie_current.txt ] && [ ! -s hippie_063.tsv ]; then
    awk -F'\t' '$5+0 >= 0.63 && $1 != "" && $3 != "" {print $1"\t"$3}' \
        hippie_current.txt > hippie_063.tsv
    echo "hippie_063.tsv  $(wc -l < hippie_063.tsv) rows"
fi

# BioGRID tab3:
#   2,3 Entrez Gene A/B   13 Experimental System Type   16,17 organism ids
#   24 SWISS-PROT A   25 TREMBL A   27 SWISS-PROT B
# Entrez rather than SWISS-PROT because it is far more complete.  Column 25 is
# TrEMBL for interactor A, not SWISS-PROT for B -- the same mistake as above,
# and the reason the first BioGRID extraction came out with max degree 3.  Both
# endpoints are constrained to human, since the per-organism file still carries
# cross-species interactions.
if [ -s biogrid_tab3.zip ] && [ ! -s biogrid_human.tsv ]; then
    unzip -o -q biogrid_tab3.zip 'BIOGRID-ORGANISM-Homo_sapiens-*.tab3.txt' \
        -d biogrid_unpacked 2>/dev/null
    f=$(ls biogrid_unpacked/BIOGRID-ORGANISM-Homo_sapiens-*.tab3.txt 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        awk -F'\t' 'NR>1 && $13=="physical" && $16=="9606" && $17=="9606" \
                    && $2!="-" && $3!="-" && $2!=$3 {print $2"\t"$3}' \
            "$f" > biogrid_human.tsv
        echo "biogrid_human.tsv  $(wc -l < biogrid_human.tsv) rows"
    fi
fi

# ---- sanity check ----------------------------------------------------------
# A real PPI network is heavy-tailed: thousands of nodes and a maximum degree in
# the hundreds.  An extraction that pairs a protein with its own alternate
# identifier instead produces a near-perfect matching -- almost every node at
# degree 1 and a maximum in single digits -- and that passes every other check,
# runs to completion, and yields a whole grid of plausible-looking near-zeros.
# It cost sixteen cluster tasks before anyone looked at a degree distribution.
echo
echo "edge lists present:"
for f in *.tsv; do
    [ -s "$f" ] || continue
    awk -F'\t' -v name="$f" '
        {d[$1]++; d[$2]++; e++}
        END {
            n = 0; mx = 0; one = 0
            for (k in d) { n++; if (d[k] > mx) mx = d[k]; if (d[k] == 1) one++ }
            printf "  %-22s %7d nodes  %9d rows  max degree %5d  deg1 %5.1f%%",
                   name, n, e, mx, 100 * one / n
            if (mx < 20 || one / n > 0.8)
                printf "   <-- IMPLAUSIBLE, check the column indices"
            printf "\n"
        }' "$f"
done
