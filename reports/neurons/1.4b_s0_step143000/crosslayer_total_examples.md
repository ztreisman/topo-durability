# Cross-layer fork/rejoin loops (total graph): longest-span chordless pure diamonds among the top cycles

(Individual loops are not tested against a null; see the typology table in crosslayer.md.)

Root = lowest neuron, target = highest; the two arms are listed in cycle order. direct_pct: percentile of the root's direct residual weight onto the target (low = only via the arms).

## span 23 (L0 -> L23), 9 neurons, arms 2+7 edges, lifetime 0.230, direct_pct 0.82, arm_pct 1.00, tok_share 1.71
- L0 n4052 (root) r2_tok 1.00: 'ators', 'ups', 'ations', 'izations', 'ists', 'ions', 'als', 'lers'
- L18 n4442 r2_tok 0.51: ' dell', ' plaintiffs', ' Citizens', ' appellants', ' Workers', ' defendants', ' Jesus', ' Brothers'
- L21 n4002 r2_tok 0.24: ' dell', ' Workers', ' Citizens', ' CAN', 'izens', ' ain', ' Veterans', ' DON'
- L22 n4464 r2_tok 0.12: ' plaintiffs', ' appellants', ' Plaintiffs', ' defendants', ' Defendants', ' dell', ' Workers', ' taxpayers'
- L22 n6145 r2_tok 0.06: "('/", " '/", " '-", 'ocomplete', 'baz', " '#", 'undefined', " '@"
- L22 n4866 r2_tok 0.13: "('/", " '-", " '/", 'undefined', " '#", " '*", " '@", "('#"
- L22 n6746 r2_tok 0.17: '))**(-', ' (-', ' ((-', ')**(-', '))/((-', ')/((-', '**(-', ')**('
- L23 n6422 (target) r2_tok 0.29: ' ((-', ')**(-', '*(-', '))**(-', ' (-', '))/((-', ')/((-', '))**('
- L20 n1115 r2_tok 0.45: ' rebounds', ' blades', 'Figs', ' Figs', ' yards', 'Conclusions', ' screens', ' replicates'

## span 22 (L1 -> L23), 10 neurons, arms 2+8 edges, lifetime 0.165, direct_pct 0.78, arm_pct 0.93, tok_share 1.43
- L1 n7164 (root) r2_tok 0.92: '<|endoftext|>', "('<", "('./", '([]', '([', '("#', "('#", '("/'
- L3 n4815 r2_tok 0.25: '<|endoftext|>', ' \n\n', '\n\t\t\t\t\t\t\t\t', '\r\n\t\t\t\t\t', '\xa0\n', '\n', 'ss', '\n\t\n'
- L23 n6473 (target) r2_tok 0.17: 'Jy', ' na', ' fistula', '<|endoftext|>', ' acqu', ' leakage', ' holes', ' Intr'
- L16 n5623 r2_tok 0.22: ' arrives', ' centrifuged', ' differentiable', ' resigned', ' compete', ' remanded', ' sits', ' admissible'
- L8 n422 r2_tok 0.30: ' incubated', ' supplemented', ' resuspended', ' centered', ' residing', ' originating', ' cultured', ' located'
- L2 n1022 r2_tok 0.59: ' configured', ' determined', ' populated', ' characterized', ' drawn', 'prepared', ' bored', ' honored'
- L2 n4349 r2_tok 0.20: ' supposed', 'し', ' divided', ' going', ' portrayed', ' subjected', ' multiplied', ' grouped'
- L2 n2695 r2_tok 0.14: 'agu', 'ichlor', ' externally', 'osin', 'odi', ' slightly', ' quantitatively', 'adem'
- L2 n7838 r2_tok 0.11: 'onz', 'itzer', ' regard', 'agu', ',{{\\', 'τη', 'iously', 'enes'
- L2 n8053 r2_tok 0.24: '<|endoftext|>', '=========', 'uren', '\x80', ' With', 'ibular', '================================================================', ' Through'

## span 22 (L1 -> L23), 7 neurons, arms 3+4 edges, lifetime 0.220, direct_pct 0.58, arm_pct 0.87, tok_share 1.82
- L1 n4654 (root) r2_tok 0.52: 'jj', ' q', ' tt', 'dq', 'pq', 'psi', 'vu', 'xf'
- L9 n4842 r2_tok 0.68: ' d', ' s', ' b', ' n', ' p', ' t', ' i', ' c'
- L22 n704 r2_tok 0.36: 'wm', 'jt', 'Ur', 'yc', 'vl', 'Ah', 'gc', 'vr'
- L22 n3870 r2_tok 0.35: 'Let', ')**(-', 'ooo', ')**(', 'Determine', ' ((-', '))**(', ' Let'
- L23 n2230 (target) r2_tok 0.33: '))/((-', ' ((-', ')/((-', ' (-', ')*(-', '*(-', 'Suppose', 'Solve'
- L15 n4120 r2_tok 0.18: 'Rearrange', 'Let', '\\](', 'Express', 'Simplify', 'Suppose', '∗∗', ' complic'
- L10 n7775 r2_tok 0.24: 'xd', ' shortly', ' Until', ' lately', ' Later', '))*-', ' Jennifer', ' soon'

## span 21 (L0 -> L21), 10 neurons, arms 4+6 edges, lifetime 0.156, direct_pct 1.00, arm_pct 0.95, tok_share 1.98
- L0 n2104 (root) r2_tok 1.00: ' Family', ' Work', ' Current', ' Character', ' Values', ' Future', ' Control', ' Collect'
- L2 n6702 r2_tok 0.68: ' Recall', ' Further', ' Having', ' Considering', ' Following', ' Based', ' Specifically', ' Observe'
- L7 n3047 r2_tok 0.41: ' Meanwhile', 'Meanwhile', ' Likewise', ' Indeed', ' Interestingly', ' Actually', 'Indeed', ' Similarly'
- L9 n3830 r2_tok 0.38: ' Specifically', ' Moreover', ' Also', ' Furthermore', ' Particularly', ' Lastly', ' Additionally', ' Finally'
- L9 n2392 r2_tok 0.35: '_:', '"}).', '"}**).', '"}.', ']\\].', ']^.', '`:', ']).'
- L15 n3148 r2_tok 0.41: ' WHETHER', ' YOUR', ' THE', '**).', ' THIS', ' WITH', ' Historic', ' OF'
- L21 n3151 (target) r2_tok 0.37: ' Related', ' Amended', ' Appellate', ' Assistant', ' Public', ' Dismiss', ' Corporate', ' Criminal'
- L11 n2806 r2_tok 0.19: ' Historic', ' Assay', ' Contributions', ' Invention', ' Related', ' Laid', ' Costs', ' Programs'
- L2 n4147 r2_tok 0.27: ' Programs', ' Dismiss', 'Metrics', ' Offer', ' Costs', ' Solutions', ' Edition', ' Places'
- L1 n4081 r2_tok 0.72: ' Lit', ' Song', ' Pharm', ' Agg', ' Sty', ' Lik', 'TIF', ' Poss'

## span 21 (L2 -> L23), 8 neurons, arms 3+5 edges, lifetime 0.189, direct_pct 0.77, arm_pct 0.86, tok_share 1.57
- L2 n7490 (root) r2_tok 0.39: ' veteran', ' graduates', ' sons', ' businessman', ' citizen', ' scientist', ' professor', ' politician'
- L4 n2625 r2_tok 0.16: ' Champion', ' footballer', ' founder', ' photographer', ' musician', ' champion', ' veteran', ' journalist'
- L14 n8012 r2_tok 0.08: ' founder', ' elected', ' chairman', ' director', ' Vice', ' appointed', ' professor', ' footballer'
- L17 n2334 r2_tok 0.09: ' Vice', ' Comiss', 'Former', ' youngest', ' former', ' Senior', ' Chief', ' Assistant'
- L18 n2976 r2_tok 0.09: ' youngest', ' educate', ' convince', ' inspire', ' dear', ' encourage', ' motivate', 'Hey'
- L23 n4641 (target) r2_tok 0.13: ' pessoas', ' Presidente', ' του', ' footballer', ' poet', ' woman', ' servant', ' Dios'
- L17 n4680 r2_tok 0.34: ' servant', ' guy', ' boyfriend', ' boy', ' grandfather', ' father', ' brother', ' stranger'
- L4 n2722 r2_tok 0.57: 'ologist', ' colleague', 'ologists', 'ographer', ' riders', ' instructor', ' journalists', 'ician'

## span 20 (L3 -> L23), 12 neurons, arms 1+11 edges, lifetime 0.262, direct_pct 0.99, arm_pct 0.96, tok_share 1.29
- L3 n1876 (root) r2_tok 0.24: '<|endoftext|>', ' \n\n', '\n\t\t\t\t\t\t\t\t', '\r\n\t\t\t\t\t', '\xa0\n', '\n', '\n\t\n', 'ss'
- L23 n922 (target) r2_tok 0.19: '<|endoftext|>', ' cannot', ' never', ' wont', 'cannot', "'t", 'without', ' not'
- L19 n2584 r2_tok 0.10: ' never', 'Anyone', ' Anyone', ' wont', ' hardly', ' Without', ' ARISING', ' without'
- L12 n3162 r2_tok 0.45: ' hardly', 'Nothing', ' never', ' dont', ' nobody', ' wont', ' nothing', ' nowhere'
- L9 n6959 r2_tok 0.25: 'ifferent', ' lacked', ' não', ' nond', ' less', ' uncon', ' rebut', ' fewer'
- L9 n5421 r2_tok 0.10: 'notin', ' refused', ' refuses', ' unwilling', ' δεν', '不', ' excluded', ' dont'
- L9 n6509 r2_tok 0.02: ' Otherwise', 'Female', ' remaining', ' unaffected', ' counterparts', 'otherwise', ' otherwise', '≥'
- L9 n1456 r2_tok 0.02: ' dst', ' rhs', ' cos', 'Height', ' placebo', ' ↑', 'Female', 'dst'
- L9 n6736 r2_tok 0.04: 'Padding', 'Height', 'trunc', 'Female', '~^', ' Packet', 'Maximum', 'Inner'
- L6 n3045 r2_tok 0.07: 'pmod', 'matches', 'region', ' rand', ' idx', ' np', 'MAG', ' keyword'
- L6 n7316 r2_tok 0.13: 'aaaaaaaa', '::::', 'mk', 'jl', 'IAM', 'irk', 'Complex', 'compiler'
- L6 n3363 r2_tok 0.12: '<|endoftext|>', '\n\t\t\t\t\t\t\t\t', '\xa0\n', '\r\n\t\t\t\t\t', '\n\t\n', ' idx', '\r\n\t\t\t\t', '::::'

## span 20 (L2 -> L22), 8 neurons, arms 3+5 edges, lifetime 0.235, direct_pct 0.24, arm_pct 0.93, tok_share 1.71
- L2 n6152 (root) r2_tok 0.34: ' CRT', ' PAS', ' BBB', ' NPR', 'PBS', ' PLC', ' DW', ' TW'
- L7 n2871 r2_tok 0.86: ' CLE', ' WILL', 'BAS', ' IMP', ' PER', ' MID', ' CAR', ' BAS'
- L7 n879 r2_tok 0.47: 'LEY', 'MAN', 'IVER', ' COUNTY', 'SON', 'TON', 'ISON', 'DISCUSSION'
- L22 n6376 (target) r2_tok 0.32: ' LIMITED', ' GENERATED', ' SOFTWARE', ' FILE', ' COMMAND', ' WITHOUT', ' LIABLE', ' THIS'
- L21 n374 r2_tok 0.20: ' COMMAND', ' THAT', ' YOUR', ' τις', 'THE', ' WITH', 'IBLE', ' δεν'
- L19 n5225 r2_tok 0.12: '+/-', ' +/-', 'Round', 'Multiply', 'Does', ' ±', 'Let', 'Determine'
- L16 n7659 r2_tok 0.17: '}}}^', 'Let', ' dps', '}}^', ' Invention', 'What', 'Collect', ')*-'
- L5 n3836 r2_tok 0.16: 'annel', ' marginal', 'substituted', 'sided', 'ichlor', '**', ' tagged', ' q'

## span 18 (L5 -> L23), 9 neurons, arms 3+6 edges, lifetime 0.235, direct_pct 0.27, arm_pct 1.00, tok_share 1.41
- L5 n1992 (root) r2_tok 0.25: ' esper', ' últ', '�', '�', ' verd', '�', ' cu', ' nue'
- L7 n7466 r2_tok 0.26: ' realiz', ' aument', ' efect', ' redu', ' constitu', ' objet', ' introdu', ' cres'
- L8 n3395 r2_tok 0.20: ' funcion', ' aument', 'ç', 'resent', 'ê', ' apo', 'ô', ' cri'
- L9 n2317 r2_tok 0.21: ' promul', 'ζ', 'resent', ' consegu', ' garant', ' ace', ' constru', ' recon'
- L12 n3775 r2_tok 0.18: ' pod', 'ζ', 'ίν', ' poder', ' signific', ' constitu', 'ρέ', ' culmin'
- L20 n2305 r2_tok 0.12: ' signific', ' constitu', 'ίν', ' contin', ' diz', ' appoint', 'resent', ' implic'
- L23 n3112 (target) r2_tok 0.17: ' não', ' quando', ' porque', ' como', 'ない', ' parece', ' isso', ' são'
- L15 n1617 r2_tok 0.47: ' porque', ' quando', ' há', ' pode', 'êm', 'emos', ' também', ' fazer'
- L12 n955 r2_tok 0.35: 'onz', ' nue', 'ñ', ' alg', ' José', ' muit', ' pró', 'ál'

## span 17 (L0 -> L17), 8 neurons, arms 3+5 edges, lifetime 0.236, direct_pct 0.44, arm_pct 0.98, tok_share 2.15
- L0 n4052 (root) r2_tok 1.00: 'ators', 'ups', 'ations', 'izations', 'ists', 'ions', 'als', 'lers'
- L11 n5585 r2_tok 0.62: 'They', ' They', 'they', ' they', ' são', ' them', 'Their', ' Defendants'
- L13 n3559 r2_tok 0.50: 'They', ' They', 'they', ' they', 'You', 'We', ' You', ' We'
- L16 n7092 r2_tok 0.55: ' we', 'We', ' We', ' she', 'They', ' I', ' he', ' who'
- L16 n7315 r2_tok 0.27: ' quando', ' já', 'Acknowledgements', ' porque', 'Below', ' signific', ' también', ' esper'
- L17 n5967 (target) r2_tok 0.17: "'ve", ' lysates', ' Which', ' já', ' supernatant', ' dawn', ' she', ' has'
- L14 n191 r2_tok 0.04: 'uents', ' surname', ' lysates', ' seedlings', ' SOFTWARE', ' mutants', 'kDa', ' affidavits'
- L14 n4811 r2_tok 0.34: 'itos', ' dps', ' Programs', 'templates', ' anos', 'ações', 'ceptors', 'idades'

## span 16 (L7 -> L23), 8 neurons, arms 3+5 edges, lifetime 0.237, direct_pct 0.54, arm_pct 0.89, tok_share 1.84
- L7 n5009 (root) r2_tok 0.33: ' declaring', ' concluding', ' testified', ' believing', ' concludes', ' hypothesized', ' assume', ' contend'
- L14 n4930 r2_tok 0.41: ' speculate', ' noting', ' insisted', ' hypothesized', ' conclude', ' argues', ' concluded', ' contend'
- L14 n2172 r2_tok 0.39: ' demonstrates', ' knew', ' knows', ' illustrates', ' demonstrate', ' know', ' realize', ' understands'
- L16 n2181 r2_tok 0.18: ' wrt', ' assuming', "''(", '/((', "'(", ' Determine', ' ++', ')_{'
- L22 n5405 r2_tok 0.30: 'Collect', 'doibase', '}}^{(', ' desde', "'(", "''(", ' até', ' durante'
- L23 n598 (target) r2_tok 0.62: ' eliminates', ' elucidate', ' avoids', ' facilitates', ' explores', ' avoiding', ' Applying', ' enables'
- L20 n924 r2_tok 0.26: ' stating', ' comprises', ' comprising', ' includes', ' alleging', ' obtains', ' Define', ' include'
- L13 n666 r2_tok 0.42: ' Assume', ' namely', 'Assume', ' noting', ' suppose', ' contend', ' contends', ' Hence'

## span 16 (L6 -> L22), 9 neurons, arms 2+7 edges, lifetime 0.193, direct_pct 1.00, arm_pct 0.91, tok_share 1.35
- L6 n5984 (root) r2_tok 0.24: 'IAM', 'olph', ' Harold', ' Marc', ' José', ' Vladimir', ' Carl', ' Jean'
- L9 n1616 r2_tok 0.20: ' Marc', ' Ian', ' Nathan', ' Jean', ' Jim', ' Tak', ' Christopher', ' José'
- L10 n5525 r2_tok 0.15: ' McD', ' McC', ' McK', ' McG', ' McL', ' Vander', 'Mah', ' McKin'
- L10 n7353 r2_tok 0.19: 'Wang', '[@', 'Hu', ' Wang', ',[@', ' Zhou', 'nik', 'Lee'
- L14 n5426 r2_tok 0.11: '[@', 'Wang', '];', 'Hu', '*.,', ' @', 'heng', '"};'
- L15 n5824 r2_tok 0.12: 'quoting', 'citing', ' quoting', ' Presidente', ' See', '.’”', '.\'"', ' citing'
- L16 n112 r2_tok 0.12: 'ñor', ' Steve', ' Barack', ' Chris', ' Andy', 'Chris', ' Randy', ' Hillary'
- L22 n2567 (target) r2_tok 0.21: ' Jennifer', ' Harold', ' Steven', 'Jeff', ' Stephen', ' David', ' Jeffrey', ' Ian'
- L12 n858 r2_tok 0.28: ' Sean', ' Timothy', ' Donald', ' Stephen', ' Keith', 'Jon', 'IAM', 'Michael'

## span 16 (L7 -> L23), 10 neurons, arms 4+6 edges, lifetime 0.158, direct_pct 0.68, arm_pct 0.99, tok_share 1.40
- L7 n4089 (root) r2_tok 0.07: ']:', ' \\**', ' \\*', ' \\*\\*', '\\**', ' \\*\\*\\*', '†', 'isks'
- L11 n3272 r2_tok 0.07: ' \\*', ' \\**', 'isks', ' \\*\\*', ' visualized', 'Statistical', '\\**', 'Note'
- L13 n7297 r2_tok 0.10: ' \\**', ' **(', ' replicates', ' (**', 'isks', ' triplicate', ' denoted', 'indicated'
- L13 n1430 r2_tok 0.11: ' Quantification', ' SEM', 'Affirmed', ' incubated', ' cytometry', '--------------------------------------------------------------------------------------------------------------------------------', ' transfected', ' (%)'
- L23 n633 (target) r2_tok 0.21: ' \\*\\*\\*', 'ivariable', '^−/−^', '*\\<', 'nM', ' CI', ' \\*\\*', 'adjusted'
- L22 n4680 r2_tok 0.23: ' +/-', 'µg', ' IgG', '+/-', 'mmol', ' nmol', ' µg', ' ±'
- L22 n2574 r2_tok 0.36: 'Let', 'mathcal', 'mathbb', ' Exhibit', 'mathbf', ' Let', 'mathscr', 'tilde'
- L18 n6347 r2_tok 0.19: 'IAM', 'mathcal', 'mathscr', ' vitamin', ' COMMAND', 'cal', ' WILL', ' Exhibit'
- L15 n909 r2_tok 0.08: 'Thanks', ' Thanks', ' thank', ' grateful', 'thanks', 'rely', ' thanks', ' acknowledge'
- L11 n6061 r2_tok 0.06: ' thank', 'Thanks', 'thanks', 'Acknowledgements', ']:', "]'", ' Thanks', 'ributed'
