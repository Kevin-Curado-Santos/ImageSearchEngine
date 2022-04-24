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
            ]  
}

np.savez_compressed("relevancy.npz", **relevancy)