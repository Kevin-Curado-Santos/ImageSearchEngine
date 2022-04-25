import numpy as np

#the key is the img query in Topic/SampleImages and the value is the list of img in the database, ImagesCLEFphoto2008/images that are 
# relevant to the query img 
relevancy = {
    
    "02/16432": ["37/37386", 
                "37/37169",
                "37/37387",
                "40/40500",
                "37/37194",
                "37/37191",
                "37/37394",
                "37/37392",
                "37/37371",
                "37/37393",
                "37/37374",
                "37/37353",
                "31/31337",
                "37/37204",
                "37/37192",
                "37/37375",
                "37/37368",
                "32/32763",
                "39/39625",
                "37/37369",
                "37/37303",
                "32/32870",
                "32/32866",
            ],
    
    "02/37395": ["37/37386", 
                "37/37169",
                "37/37387",
                "40/40500",
                "37/37194",
                "37/37191",
                "37/37394",
                "37/37392",
                "37/37371",
                "37/37393",
                "37/37374",
                "37/37353",
                "31/31337",
                "37/37204",
                "37/37192",
                "37/37375",
                "37/37368",
                "32/32763",
                "39/39625",
                "37/37369",
                "37/37303",
                "32/32870",
                "32/32866",
            ],

    "02/40498": ["37/37386", 
                "37/37169",
                "37/37387",
                "40/40500",
                "37/37194",
                "37/37191",
                "37/37394",
                "37/37392",
                "37/37371",
                "37/37393",
                "37/37374",
                "37/37353",
                "31/31337",
                "37/37204",
                "37/37192",
                "37/37375",
                "37/37368",
                "32/32763",
                "39/39625",
                "37/37369",
                "37/37303",
                "32/32870",
                "32/32866",
            ],

    "03/31":    ["04/4531",
                "04/4533",
                "30/30011",
                "00/28",
                "02/2369",
                "04/4815",
                "35/35783",
                "03/3147",
                "15/15330",
                "35/35782",
                "16/16819",
                "35/35780",
                "04/4759",
                "14/14520",
                "15/15973",
                "36/36035",
                "36/36036",
                "04/4900",
                "35/35678",
                "06/6501",
                "20/20357",
                "19/19211",
                "19/19194",
                "19/19135",
                "19/19198",
                "19/19259",
                "25/25966",
                "20/20356",
                "04/4896",
                "04/4895",
                "10/10999",
                "37/37051",
                "23/23108",
                "10/10657",
                "16/16993",
                "22/22777"
            ],
    
    "03/7446":  ["04/4531",
                "04/4533",
                "30/30011",
                "00/28",
                "02/2369",
                "04/4815",
                "35/35783",
                "03/3147",
                "15/15330",
                "35/35782",
                "16/16819",
                "35/35780",
                "04/4759",
                "14/14520",
                "15/15973",
                "36/36035",
                "36/36036",
                "04/4900",
                "35/35678",
                "06/6501",
                "20/20357",
                "19/19211",
                "19/19194",
                "19/19135",
                "19/19198",
                "19/19259",
                "25/25966",
                "20/20356",
                "04/4896",
                "04/4895",
                "10/10999",
                "37/37051",
                "23/23108",
                "10/10657",
                "16/16993",
                "22/22777"
            ],

    "03/35577": ["04/4531",
                "04/4533",
                "30/30011",
                "00/28",
                "02/2369",
                "04/4815",
                "35/35783",
                "03/3147",
                "15/15330",
                "35/35782",
                "16/16819",
                "35/35780",
                "04/4759",
                "14/14520",
                "15/15973",
                "36/36035",
                "36/36036",
                "04/4900",
                "35/35678",
                "06/6501",
                "20/20357",
                "19/19211",
                "19/19194",
                "19/19135",
                "19/19198",
                "19/19259",
                "25/25966",
                "20/20356",
                "04/4896",
                "04/4895",
                "10/10999",
                "37/37051",
                "23/23108",
                "10/10657",
                "16/16993",
                "22/22777"
            ]

}

np.savez_compressed("relevancy.npz", **relevancy)