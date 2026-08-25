# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

SOURCES={
 "contracts":"Benveniste et al., Contracts for System Design (2018); de Alfaro & Henzinger, Interface Theories for Component-Based Design (2001)",
 "platform":"Keutzer et al., System-Level Design: Orthogonalization of Concerns and Platform-Based Design (2000)",
 "vector":"Löhne, Vector Optimization with Infimum and Supremum (2011); Löhne, Rudloff & Ulus, convex vector optimization upper-image algorithms (2014)",
 "convex":"Rockafellar, Convex Analysis (1970)",
 "cc":"Kushilevitz & Nisan, Communication Complexity (1997); contextual/characteristic-graph methods in distributed function computation",
 "network":"Ahlswede et al., Network Information Flow (2000); Appuswamy/Guang/Yeung line on network function computation; Tripathy & Ramamoorthy zero-error rate regions",
 "regular":"Bingham, Goldie & Teugels, Regular Variation (1987)",
 "dsi":"Sornette, Discrete-Scale Invariance and Complex Dimensions (1998)",
 "pf":"Birkhoff (1957); Bushell (1973); Perron-Frobenius/Floquet theory for positive operators and periodic products",
 "semialg":"Bochnak, Coste & Roy, Real Algebraic Geometry (1998)",
 "dse":"Timeloop, MAESTRO, ZigZag, CoSA and Gemini design-space exploration literature",
 "io":"Hong & Kung (1981); Ballard et al. communication lower bounds; Frigo et al. cache-oblivious algorithms",
 "attention":"Dao et al., FlashAttention (2022)",
 "matching":"Classical maximum-weight bipartite assignment and Birkhoff-von Neumann matching theory",
}

# status codes:
# PASS-S: proof is standard/elementary and internally checked or directly reducible.
# PASS-C: proof is coherent and computationally corroborated, but external specialist review is still warranted.
# IMPORTED: result is explicitly imported from prior work.
# OPEN-N: correctness appears sound; novelty priority requires specialist literature audit.
# OPEN-CN: both specialist proof review and novelty priority remain open.

rows=[]
def add(id,title,page,status,novelty,relation,prior,risk,action,check="",notes=""):
 rows.append({
  "result_id":id,"result_title":title,"manuscript_page":page,"correctness_status":status,
  "novelty_class":novelty,"relation_to_existing_fields":relation,"closest_prior_art":prior,
  "risk":"High" if risk==3 else "Medium" if risk==2 else "Low",
  "recommended_action":action,"audit_check":check,"audit_notes":notes
 })

add("Def.2.2/Def.5.2","Resource contract and hierarchy-indexed CRG realization bundle","6, 14","PASS-C","Candidate new unifying object / formal synthesis","Contract-based and platform-based design already separate specifications, platforms and implementations; vector optimization already uses images and upper images. CRG's distinctive claim is the computation-first coupling of legal realization identity, typed interface obligations, hierarchy maps and technology-late observation.",SOURCES["contracts"]+"; "+SOURCES["platform"]+"; "+SOURCES["vector"],3,"Commission joint review by contract-theory, vector-optimization and EDA specialists; claim novelty in the combined operational object, not in contracts or upper images individually.","A03","The object is the main system-level novelty candidate. Mathematical subobjects are individually familiar.")
add("2.3","Functoriality of contract pushforward","6","PASS-S","New application/formalization of elementary associativity","Standard functorial pushforward of additive, max-plus and direct-image charges.",SOURCES["contracts"],1,"Retain; describe as formal infrastructure rather than a new mathematical theorem.","","Proof is a direct regrouping/associativity argument.")
add("3.3","Canonical directed-summary theorem","7-8","PASS-C / OPEN-N","Candidate new theorem formulation; strong Myhill-Nerode-style adaptation","Contextual equivalence and minimal deterministic states are established ideas; the arbitrary terminal placement, directed tree interfaces, simultaneous joint realizability and uniqueness-up-to-renaming are the potentially distinctive package.",SOURCES["cc"],2,"Seek specialist review in communication complexity, automata/Myhill-Nerode methods and distributed tree computation; narrow novelty claim to the stated multi-terminal tree model.","A01","Proof is internally coherent and exhaustively checked on a finite tree example.")
add("3.4","Fixed-bit canonical widths","8","PASS-S","Immediate corollary/new application","Ceiling semantic state cardinality to fixed-length binary width is standard.",SOURCES["cc"],1,"Retain as corollary; no standalone novelty claim.","A01","")
add("3.5","Single-root quotient","8","PASS-S","Known analogy / specialization","Myhill-Nerode-like subtree quotient for rooted computation.",SOURCES["cc"],1,"Frame as specialization and conceptual bridge.","A01","")
add("3.6","Independent-bit routing","8","PASS-S","New application of quotient theorem","Identity/copy demands reduce to separated independent bit counts.",SOURCES["cc"],1,"Retain as calibration example.","A01","")
add("4.3","Circuit-to-protocol factorization","9","PASS-S","Standard representation translation with explicit accounting","Routing each hyperedge over the minimal tree subtree is standard; equality depends on the free-forwarding contract.",SOURCES["platform"],1,"Keep explicit contract scope; do not claim a new circuit/protocol equivalence in general.","","")
add("4.5","Protocol-to-circuit compilation","10","PASS-S","Standard constructive compilation","Topologically instantiating event functions as circuits is standard.",SOURCES["cc"],1,"Retain as explicit overhead ledger.","","")
add("4.6","Representation equivalence with explicit overhead","10","PASS-S","Synthesis of 4.3/4.5","A scoped equivalence, not a new unrestricted equivalence.",SOURCES["cc"],1,"Retain; emphasize contracts and overheads.","","")
add("4.8","Linear contextual states are cut ranks","10","PASS-S","Established linear-algebraic characterization","Cosets of the kernel and image rank are standard in linear communication/function computation.",SOURCES["network"],1,"Cite prior linear-function computation explicitly; no novelty claim for the one-cut fact.","A02","")
add("4.9","Interactive directional range bound","11","PASS-S","Established transcript/range converse","Fixing receiver context and counting forward transcript states is standard deterministic communication complexity.",SOURCES["cc"],1,"State as standard lemma supporting the joint theorem.","A02","")
add("4.10","Local composition of linear quotient states","11","PASS-S","Elementary linear-algebraic lemma","Kernel inclusion under row deletion is elementary.",SOURCES["network"],1,"Retain as proof machinery.","A02","")
add("4.12","Exact interactive linear lower corner and orthant","11-12","PASS-C / OPEN-CN","Candidate new exact theorem","Prior work gives cut-rank/range converses and network-function results. The candidate novelty is simultaneous attainment of every directed tree-edge lower corner for arbitrary input/output placement, free reverse interaction and nonlinear local encoders.",SOURCES["network"]+"; "+SOURCES["cc"],3,"Priority external proof and novelty review. Search tree-network function-computation literature specifically for an equivalent simultaneous directed orthant theorem.","A02","Finite-field range/rank identity passed 40 random checks; the all-edge constructive proof was manually reviewed but not externally certified.")
add("4.13","Carrier/network-code equivalence","12","PASS-S","Definitional equivalence / established application","The local encoder/decoder objects are network function codes; block closure and normalization are conventional.",SOURCES["network"],1,"Present as bridge to prior capacity-region literature, not a new network-coding theorem.","","")
add("4.15","Cut-state conservation","13","PASS-S","Elementary counting theorem in a new typed certificate form","Receiver initial state plus forward transcript must distinguish required output states; the inequality is a counting converse.",SOURCES["cc"]+"; Hong-Kung/area-time traditions",1,"Novelty should be claimed in certificate packaging and compositional use, not the inequality itself.","","")
add("4.16","Fractional certificate packing","13","PASS-S","Standard fractional packing/LP summation","Weighted sum of valid inequalities under per-event load constraints.","Fractional packing and LP duality",1,"Retain as useful composition rule; no standalone novelty claim.","","")
add("5.3","Semantic isomorphism invariance","14-15","PASS-S","Elementary invariance","Relabeling invariance is standard.",SOURCES["contracts"],1,"Retain as hygiene property.","","")
add("5.5","Hierarchy-contraction consistency","15","PASS-C / OPEN-N","Candidate new coherence theorem for CRG object","Hierarchy contraction and quotient/abstraction maps are familiar in compositional design; the typed realization/signature inclusion is specific to CRG.",SOURCES["contracts"]+"; "+SOURCES["platform"],2,"Seek category/compositional-systems review; likely novelty is the instantiated object, not categorical machinery.","","")
add("5.6","Decomposition-system coherence","15","PASS-S / OPEN-N","New formal synthesis","Identity/composition is elementary after definitions; value lies in showing CRG is a coherent indexed system.",SOURCES["contracts"],2,"Describe as structural theorem of the new object.","","")
add("5.8","Reconstruction by monotone interface observations","15-16","PASS-S","Established order-topology idea applied to CRG","Upper-closed set membership is recovered by monotone distance probes; vector optimization already treats upper images as the natural objective-space object.",SOURCES["vector"],1,"Explicitly credit upper-image/vector-optimization lineage. Claim the CRG observation interpretation, not mathematical priority.","A03","")
add("5.10","Linear-cost-body duality","16","PASS-S","Established convex separation/support duality","Closed convex upper sets are reconstructed by nonnegative linear support prices.",SOURCES["convex"]+"; "+SOURCES["vector"],1,"Classify as imported convex-analysis theorem specialized to CRG.","A03","")
add("5.11","Linear-price completeness","16","PASS-S","Immediate corollary","Direct equivalence of K and all nonnegative linear price responses.",SOURCES["convex"],1,"No standalone novelty claim.","A03","")
add("5.12","Universal lossless retention","16","PASS-S / OPEN-N","Conceptually central new application; theorem follows from 5.8/5.10","Sufficient-statistic/upper-image preservation is familiar in optimization, but the commitment-regret interpretation and linkage to legal realization retention are distinctive CRG contributions.",SOURCES["vector"]+"; robust/multiobjective optimization",2,"Claim novelty in the exact design-retention principle and operational semantics, while acknowledging the proof is a direct reconstruction corollary.","A03","")
add("5.13","Strict insufficiency of marginal and scalar summaries","17","PASS-S","Elementary counterexample / explanatory result","Classic multiobjective fact that marginal minima and scalar summaries do not recover a Pareto image.",SOURCES["vector"],1,"Retain as pedagogical theorem; no novelty claim.","A03","")
add("5.14","Why exact region cannot be convexified universally","17","PASS-S","Elementary nonconvex counterexample","Weighted sums see convex hull; nonlinear objectives and discrete attainability do not.",SOURCES["vector"],1,"Retain to distinguish R from K.","A03","")
add("5.15","Price reversal for incomparable signatures","17","PASS-S","Elementary Pareto-order fact","Coordinate basis prices reverse incomparable points.",SOURCES["vector"],1,"Retain as concise consequence.","A03","")
add("6.1","Resource monotonicity","17","PASS-S","Immediate set-inclusion result","Relaxed contracts enlarge feasible sets.",SOURCES["contracts"],1,"No novelty claim.","","")
add("6.2","Local data processing","17","PASS-S","Standard data-processing style construction","Free local pre/post processing cannot raise interface optimum.",SOURCES["cc"],1,"Frame as CRG monotonicity analogue.","","")
add("6.3","Exact product additivity for canonical summaries","18","PASS-C / OPEN-N","Candidate new exact specialization","Product contextual equivalence factors for disjoint tasks; close to standard automata/communication product properties.",SOURCES["cc"],2,"Seek prior-art review on Myhill-Nerode product quotients and deterministic communication state complexity.","A04","")
add("6.4","Constructive subadditivity and communication synergy","18","PASS-S","Standard feasible-composition bound; new named metric","Parallel execution gives subadditivity; the price-weighted synergy definition is an CRG bookkeeping device.","Direct-sum/direct-product and information-complexity literature",1,"Do not imply a new direct-sum theorem.","","")
add("8.2","Stable-scale limit","20","PASS-S","Classical regular-variation characterization","Measurable positive dilation limits are powers and yield regular variation.",SOURCES["regular"],1,"Mark as imported theorem.","A05","")
add("8.3","Pure power as exact fixed point","20","PASS-S","Classical corollary","Exact multiplicative covariance implies a power law.",SOURCES["regular"],1,"No novelty claim.","A05","")
add("8.4","Discrete scale covariance","20","PASS-S","Classical discrete-scale-invariance form","Preferred-factor covariance yields a periodic function of log scale.",SOURCES["dsi"],1,"Cite discrete-scale-invariance literature prominently.","A05","")
add("8.5","Two-scale criterion","20","PASS-S / OPEN-N","Known-type density/monotonicity criterion; possibly new concise formulation","Irrational log-ratio generates a dense multiplicative subgroup; monotonicity squeezes all scales.",SOURCES["regular"],1,"Check whether identical criterion is in regular-variation literature; likely not a major novelty claim.","A05","")
add("8.7","Fixed points and scale cycles","21","PASS-S","Reformulation of regular/discrete scale invariance","Normalized dilation fixed points/cycles restate established equations.",SOURCES["regular"]+"; "+SOURCES["dsi"],1,"Frame as CRG dynamical language.","A05","")
add("8.8","Phase-ratio characterization","21","PASS-C / OPEN-CN","Candidate new theorem or strengthened synthesis","Phasewise adjacent-ratio convergence is converted into power × slowly varying × periodic representation under canonical log-linear interpolation.",SOURCES["regular"]+"; "+SOURCES["dsi"],3,"Priority specialist review in regular variation and discrete-scale invariance; verify all interpolation and summability clauses.","A05","Synthetic two-phase sequence passed. Proof appears coherent but priority and full rigor require specialist review.")
add("9.1","Multiplicative-window convergence","22","PASS-S","Straightforward corollary of uniform convergence for slow variation","OLS weights cancel amplitude and power terms; slow variation removes finite-window bias asymptotically.",SOURCES["regular"],1,"Retain as estimator lemma; no major novelty claim.","A05","")
add("10.1","Summable scalar distortion","23","PASS-S","Standard infinite-product argument","Absolute summability yields convergent multiplicative correction.",SOURCES["regular"],1,"No novelty claim.","A06","")
add("10.3","Perron-Floquet scaling on certified mixing class","24-25","PASS-C / OPEN-CN","Candidate new theorem / nontrivial synthesis","Combines asymptotically periodic positive cocycles, Hilbert-metric contraction, signed vanishing relative perturbations, phase gains, regular variation and a summable strengthening.",SOURCES["pf"]+"; "+SOURCES["regular"],3,"Priority external proof review by positive-operator/dynamical-systems specialist; search random/nonautonomous Perron-Frobenius and Floquet literature for equivalent results.","A06","Numerical adversarial test passed; numerical corroboration does not certify the full proof.")
add("10.4","Stationary primitive recursion","25","PASS-S","Standard stationary specialization","Primitive positive matrix plus vanishing perturbation yields Perron direction and growth rate.",SOURCES["pf"],1,"Treat as corollary.","A06","")
add("11.1","Mode-mixture geometry","26","PASS-S","Standard calculus identity applied to traffic modes","Log-slope is share-weighted mean; curvature decomposes into between-mode variance plus within-mode drift.","Mixture calculus / log-sum-exp derivatives",1,"Novelty lies in CRG attribution use, not identity.","A07","")
add("11.2","No exact heterogeneous scalar","26","PASS-S","Immediate corollary","Positive sum of unequal powers has nonzero log curvature.",SOURCES["regular"],1,"Retain as interpretation.","A07","")
add("11.3","Dominant regularly varying mode","26","PASS-S","Classical closure property","Finite positive sum takes the maximum regular-variation index.",SOURCES["regular"],1,"Mark as imported/standard.","A07","")
add("12.2","Boundary crossover","28","PASS-S / OPEN-N","Natural regular-variation composition theorem","Normalized capped regularly varying profiles converge to a declared cap response; finite phase persists.",SOURCES["regular"]+"; "+SOURCES["dsi"],2,"Likely a new application/formulation; specialist check of priority but low correctness risk.","A08","")
add("12.3","Crossover slope","28","PASS-S","Straightforward chain-rule corollary","Elasticity of cap response multiplies interior index.",SOURCES["regular"],1,"Retain as corollary.","A08","")
add("12.4","Partial converse from phase-rich cap collapse","28","PASS-C / OPEN-CN","Candidate new converse","Local normalized cap collapse on dense anchors plus monotonicity is used to recover regular variation.",SOURCES["regular"],3,"External proof review; test edge cases involving hard caps, insufficient phase sampling and anchor density.","A08","Synthetic injective soft-cap case passed; full proof remains specialist-review item.")
add("13.1","Concentration of a fitted slope","29","PASS-S","Standard Hoeffding/union-bound application","Bounded independent contributors concentrate log-level means and OLS slope.","Hoeffding concentration and OLS linear functionals",1,"No novelty claim beyond application.","","")
add("13.2","Stable empirical generalized law","30","PASS-S / OPEN-N","New synthesis/corollary","Combines slow-variation bias with effective-independent-composition stochastic width.",SOURCES["regular"],2,"Present as conditional synthesis, not universal empirical law.","","")
add("15.2","Signature dominance","37","PASS-S","Standard Pareto monotonicity","Dominated signatures cannot win a monotone objective under common embedding conditions.",SOURCES["vector"],1,"No novelty claim.","","")
add("15.4","Finite technology-phase stratification","38","PASS-S","Standard semialgebraic cell-decomposition application","Finite rational/min-max registry labels are constant on sign-invariant semialgebraic cells.",SOURCES["semialg"],1,"Credit real algebraic geometry; novelty is the complete CRG label set.","A09","")
add("15.5","Certified full-label phase radius","38","PASS-S","Elementary Lipschitz sign-preservation bound","Distance to all sign walls bounds label stability.","Lipschitz robustness/margin analysis",1,"Retain as useful certificate.","A09","")
add("15.7","Certified sandwich and safe pruning","39","PASS-S","Standard branch-and-bound/relaxation principle","Optimistic outer relaxation is a lower bound; constructive witness is an upper bound.",SOURCES["vector"]+"; "+SOURCES["dse"],1,"Novelty claim should concern dependency-scoped integration, not sandwich logic.","A09","")
add("15.8","Argmin-hitting retention","39","PASS-S","Elementary finite-set criterion; new application","A retained set preserves a finite technology ledger iff it intersects each argmin set.","Robust optimization/scenario covering",1,"Retain as exact finite-ledger statistic.","A09","")
add("16.1","Scaling strain","39","PASS-S","Classical quotient closure of regular variation","Demand/service ratio has index p-q.",SOURCES["regular"],1,"No mathematical novelty claim; emphasize architecture-technology interpretation.","A08","")
add("16.2","Homogeneous dimensional threshold","40","PASS-S","Application of surface/volume scaling","Ideal d-dimensional boundary service index is (d-1)/d.","Classical separator/surface scaling; Thompson VLSI theory",1,"Keep scoped; not a routability or 3D superiority theorem.","A08","")
add("16.3","Aligned-link feasibility and serialization","40","PASS-S","Elementary model identity","Independent links give max demand/capacity serialization.","Network calculus/link budgeting",1,"Retain as declared fluid-link model.","","")
add("16.4","Chiplet link-budget law","40-41","PASS-S","Elementary engineering lower bounds","Bandwidth gives lane count; per-bit energy gives energy/power; shoreline areas add.","Package link budgeting and UCIe capability models",1,"Use as bridge to physical study; not signoff.","","")
add("18.1","Multilevel matrix-multiplication bracket","41-42","PASS-S / IMPORTED","Known lower/upper bounds composed across inclusive ideal-cache levels","Two-level communication lower bounds and cache-oblivious simultaneous upper bounds are established.",SOURCES["io"],1,"Classify as CRG registration of known results, not new matrix-multiplication theorem.","","")
add("18.2","Imported exact-attention I/O bracket","43","IMPORTED","Imported theorem","Directly attributed to FlashAttention analysis.",SOURCES["attention"],1,"Keep clearly marked imported.","","")
add("18.3","Static provisioning and time-shared state","44","PASS-S","Elementary resource-aggregation distinction","Simultaneous dedicated state adds; disjoint time-shared state takes maximum.","Scheduling/resource provisioning",1,"Retain as contract clarification.","","")
add("18.4","Exact aligned-layout assignment","45","PASS-C / OPEN-N","Known assignment formulation plus candidate new closed-form application","Maximum retained overlap is a standard assignment problem; nested-factor closed form and CRG ownership interpretation may be new application.",SOURCES["matching"],2,"Cite matching theory; claim novelty only for the ownership-layout specialization and closed-form conditions.","A10","")

# Write files
fields=list(rows[0].keys())
with open(ROOT/'theorem_audit_matrix.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)
(ROOT/'theorem_audit_matrix.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

# Summary counts
from collections import Counter
summary={
 "result_count":len(rows),
 "correctness_status_counts":dict(Counter(r['correctness_status'] for r in rows)),
 "novelty_class_counts":dict(Counter(r['novelty_class'] for r in rows)),
 "risk_counts":dict(Counter(r['risk'] for r in rows)),
 "high_priority_results":[r['result_id'] for r in rows if r['risk']=='High'],
}
(ROOT/'audit_matrix_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
