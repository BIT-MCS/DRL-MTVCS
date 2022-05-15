class Setting(object):
    def __init__(self, log):
        self.V = {
            'MAP_X': 16,
            'MAP_Y': 16,
            'MAX_VALUE': 1.,
            'MIN_VALUE': 0.,
            'DATA': [[1.0324544460e-01, 3.9710518718e-01, 3.1816589832e-01],
                     [6.4994609356e-01, 8.7086403370e-01, 3.8021117449e-01],
                     [6.3411630690e-02, 4.7101622820e-01, 1.2663589418e-01],
                     [2.0109863579e-01, 2.1954549849e-01, 9.0065014362e-01],
                     [5.5664062500e-01, 5.1367187500e-01, 8.5888671875e-01],
                     [2.7721959352e-01, 8.5714942217e-01, 7.5913250446e-01],
                     [5.9475517273e-01, 3.5107761621e-01, 9.5117104053e-01],
                     [3.9324176311e-01, 9.6922457218e-02, 8.2653641701e-01],
                     [6.2890625000e-01, 6.3049316406e-02, 1.4221191406e-02],
                     [2.9708996415e-01, 8.3021926880e-01, 9.7169095278e-01],
                     [8.3945250511e-01, 7.1156030893e-01, 4.3612401932e-02],
                     [7.5895112753e-01, 8.5024994612e-01, 9.0238118172e-01],
                     [9.1281473637e-01, 9.5487010479e-01, 6.4860934019e-01],
                     [8.9404296875e-01, 5.6121826172e-02, 5.8984375000e-01],
                     [9.8536413908e-01, 2.0069421828e-01, 7.8445029259e-01],
                     [9.8046875000e-01, 6.2402343750e-01, 7.6513671875e-01],
                     [6.4975386858e-01, 4.6757587790e-01, 5.2902477980e-01],
                     [7.8963375092e-01, 9.1148948669e-01, 2.1709716320e-01],
                     [9.3153107166e-01, 6.1284852028e-01, 1.8491835892e-01],
                     [3.5034179688e-01, 8.5742187500e-01, 3.8378906250e-01],
                     [1.9073486328e-02, 8.3984375000e-01, 1.1065673828e-01],
                     [9.2659670115e-01, 6.3110172749e-01, 1.9751177728e-01],
                     [9.9317830801e-01, 9.2116516829e-01, 9.8397827148e-01],
                     [2.7535066009e-01, 5.0247031450e-01, 8.0501943827e-01],
                     [7.0273977518e-01, 9.4784998894e-01, 9.2675030231e-01],
                     [9.3212890625e-01, 1.2121582031e-01, 3.5668945312e-01],
                     [7.0751953125e-01, 8.8671875000e-01, 3.8037109375e-01],
                     [8.5021108389e-02, 7.8108876944e-01, 9.6387892962e-01],
                     [3.4507930279e-01, 6.7364865541e-01, 3.7991678715e-01],
                     [9.8158401251e-01, 3.0603060126e-01, 9.5439594984e-01],
                     [3.2957059145e-01, 8.5441333055e-01, 4.1537740827e-01],
                     [4.0030962229e-01, 3.1000861526e-01, 3.2610446215e-01],
                     [2.1595486999e-01, 2.1249526180e-03, 9.8689264059e-01],
                     [9.6127945185e-01, 5.6920462847e-01, 3.4318235517e-01],
                     [4.0744027495e-01, 3.7743341923e-01, 4.5176282525e-01],
                     [9.1324138641e-01, 8.9887416363e-01, 7.6342113316e-02],
                     [3.3245229721e-01, 1.5019649267e-01, 1.8977122009e-01],
                     [6.0042887926e-01, 9.3866065145e-02, 9.8773932457e-01],
                     [6.0489833355e-01, 3.8460022211e-01, 1.2779636681e-01],
                     [2.9479980469e-02, 4.4897460938e-01, 3.2592773438e-01],
                     [6.1933040619e-01, 7.2026625276e-02, 7.0134717226e-01],
                     [4.1491749883e-01, 7.0196050406e-01, 9.7920680046e-01],
                     [6.6148382425e-01, 4.1362029314e-01, 8.3709126711e-01],
                     [9.3896484375e-01, 3.7060546875e-01, 6.0546875000e-01],
                     [5.0082415342e-01, 4.2516428232e-01, 1.2075330317e-01],
                     [9.8730468750e-01, 8.6181640625e-01, 6.8505859375e-01],
                     [8.1473779678e-01, 9.9157887697e-01, 8.4824615717e-01],
                     [7.6772552729e-01, 3.9482223988e-01, 1.1494731158e-01],
                     [2.2180175781e-01, 8.1494140625e-01, 1.0000000000e+00],
                     [9.6068486571e-02, 5.7112270594e-01, 9.5267012715e-02],
                     [2.8709378839e-01, 5.5973161012e-02, 9.2548376322e-01],
                     [2.9394531250e-01, 6.6406250000e-01, 8.8964843750e-01],
                     [1.0155130178e-01, 6.1640030146e-01, 9.3228501081e-01],
                     [7.0654296875e-01, 5.6091308594e-02, 2.2302246094e-01],
                     [3.4545898438e-01, 3.2812500000e-01, 3.4033203125e-01],
                     [8.9765793085e-01, 9.5333093405e-01, 6.1508572102e-01],
                     [6.2792968750e-01, 1.0070800781e-01, 3.1494140625e-01],
                     [6.7983239889e-01, 1.6535723209e-01, 1.3628305495e-01],
                     [9.8632812500e-01, 3.0786132812e-01, 6.1328125000e-01],
                     [4.0536481142e-01, 2.2970201075e-01, 8.8868665695e-01],
                     [7.5244128704e-01, 5.5036652088e-01, 8.0360203981e-01],
                     [4.2377990484e-01, 4.9684226513e-01, 4.9871894717e-01],
                     [9.3994140625e-01, 2.4902343750e-01, 4.0063476562e-01],
                     [2.7612304688e-01, 7.6269531250e-01, 3.9843750000e-01],
                     [7.0338428020e-02, 1.8416030705e-01, 2.7972826362e-01],
                     [4.1875004768e-02, 5.6838881969e-01, 9.1200232506e-01],
                     [7.4466329813e-01, 7.9355657101e-01, 5.7283379138e-02],
                     [7.0375454426e-01, 1.0121350735e-01, 6.0301905870e-01],
                     [7.3828125000e-01, 6.7919921875e-01, 1.3439941406e-01],
                     [5.9469288588e-01, 4.2133506387e-02, 1.5759619419e-03],
                     [2.7058112621e-01, 3.9126634598e-01, 7.1686014533e-02],
                     [6.6888642311e-01, 1.7548845708e-01, 4.0762349963e-02],
                     [8.6514510214e-02, 3.2839676738e-01, 5.4247480631e-01],
                     [8.2067781687e-01, 7.0492261648e-01, 7.6626884937e-01],
                     [4.6704962850e-01, 5.6792676449e-01, 7.8147149086e-01],
                     [2.6954671741e-01, 7.4662037194e-02, 6.4390140772e-01],
                     [6.9603852928e-02, 4.3899068236e-01, 6.8140894175e-01],
                     [2.2711746395e-01, 9.0340918303e-01, 1.7526498064e-02],
                     [5.1923253341e-04, 6.4975965023e-01, 2.9899457097e-01],
                     [4.1640073061e-01, 1.2030874193e-01, 3.3343014121e-01],
                     [5.3816523403e-02, 4.1089639068e-01, 2.0788182318e-01],
                     [6.9287109375e-01, 3.9843750000e-01, 4.0258789062e-01],
                     [5.3868941905e-05, 8.5712313652e-01, 4.4344443083e-01],
                     [7.3854434490e-01, 2.3033763282e-03, 4.6510556340e-01],
                     [5.3366171196e-03, 6.5480089188e-01, 6.1265259981e-01],
                     [5.9915184975e-01, 9.8832309246e-02, 3.8878601044e-02],
                     [8.9272898436e-01, 2.4017582834e-01, 2.6079878211e-01],
                     [4.4682543725e-02, 6.5594929457e-01, 3.6382278800e-01],
                     [2.8482139111e-01, 5.2063441277e-01, 4.0502789617e-01],
                     [6.6501933336e-01, 5.3298896551e-01, 7.2510921955e-01],
                     [9.4279795885e-01, 8.8894617558e-01, 9.6032805741e-02],
                     [2.7058202028e-01, 6.8111360073e-01, 6.7908871174e-01],
                     [3.3593311906e-01, 6.3600528240e-01, 1.3301706314e-01],
                     [8.1480801105e-02, 1.2220227718e-01, 8.2829052210e-01],
                     [6.5342682600e-01, 2.9977444559e-02, 4.0081925690e-02],
                     [1.0574880987e-01, 3.9363932610e-01, 6.3385343552e-01],
                     [7.2429925203e-01, 2.1268381178e-01, 6.6593706608e-01],
                     [7.6656169258e-03, 4.9032399058e-01, 6.6143393517e-01],
                     [3.5709509254e-01, 3.3214753866e-01, 8.7807601690e-01],
                     [8.9445441961e-01, 4.0942943096e-01, 2.9583504796e-01],
                     [2.0875360072e-01, 3.7320208549e-01, 8.7561237812e-01],
                     [5.5185019970e-01, 7.7914661169e-01, 1.6010947526e-01],
                     [9.0576171875e-02, 5.9936523438e-02, 3.7841796875e-01],
                     [9.1025221348e-01, 1.3349747285e-02, 2.2209104896e-01],
                     [2.6657441258e-01, 6.9344896078e-01, 2.6074582338e-01],
                     [7.3020601273e-01, 5.5518722534e-01, 5.1010549068e-01],
                     [2.4826626480e-01, 1.8795210123e-01, 3.8500887156e-01],
                     [6.4257812500e-01, 6.5917968750e-01, 9.2333984375e-01],
                     [8.9738041162e-01, 2.3100611567e-01, 9.3366050720e-01],
                     [4.1357731819e-01, 7.7049511671e-01, 9.6929973364e-01],
                     [5.4678139277e-03, 5.8677083254e-01, 6.6218060255e-01],
                     [7.4912065268e-01, 4.8197910190e-01, 6.5818357468e-01],
                     [1.8125968054e-02, 9.5578067005e-02, 4.4050019979e-01],
                     [5.7964795828e-01, 1.8582311273e-01, 6.2456846237e-01],
                     [7.6896733046e-01, 8.2660925388e-01, 6.6888660192e-01],
                     [6.0970419645e-01, 6.9043529034e-01, 9.1838258505e-01],
                     [9.9123787880e-01, 5.4381519556e-01, 5.7199722528e-01],
                     [3.4858676791e-01, 9.2708301544e-01, 4.2630031705e-01],
                     [3.6352717876e-01, 7.8548616171e-01, 2.9805025458e-01],
                     [2.2466266528e-03, 1.2156383693e-01, 8.5125154257e-01],
                     [9.3798828125e-01, 7.4218750000e-01, 7.5878906250e-01],
                     [9.9462890625e-01, 3.7109375000e-01, 1.5527343750e-01],
                     [9.1210937500e-01, 1.4562988281e-01, 1.3696289062e-01],
                     [2.6325130463e-01, 2.2196845710e-01, 2.7255934477e-01],
                     [9.7910113633e-02, 3.4200802445e-01, 2.2844693065e-01],
                     [5.4253208637e-01, 8.7008768320e-01, 5.7344657183e-01],
                     [5.9023308754e-01, 3.0169667676e-02, 5.4200989008e-01],
                     [7.4186164141e-01, 6.1419535428e-02, 6.1623907089e-01],
                     [7.5721389055e-01, 9.5430094004e-01, 5.7620316744e-01],
                     [5.4439667612e-02, 6.6869109869e-01, 2.6656448841e-01],
                     [3.1341722608e-01, 2.9512405396e-01, 6.1065787077e-01],
                     [3.8020551205e-01, 9.5896697044e-01, 5.7301864028e-02],
                     [3.6905708909e-01, 8.1851613522e-01, 8.7484014034e-01],
                     [1.1004638672e-01, 2.5146484375e-01, 9.9658203125e-01],
                     [6.2681418657e-01, 7.4739828706e-02, 3.9011996984e-01],
                     [2.0792730153e-01, 6.0791122913e-01, 2.8571513295e-01],
                     [4.0885549039e-02, 4.3936932087e-01, 9.8184096813e-01],
                     [4.0858674049e-01, 3.0231821537e-01, 3.2349795103e-01],
                     [5.2702003717e-01, 9.7409266233e-01, 4.9174532294e-02],
                     [2.3388645053e-01, 4.4053792953e-02, 3.2516393065e-01],
                     [7.5927007198e-01, 6.0571891069e-01, 2.6922351122e-01],
                     [5.6155454367e-02, 2.6977393031e-01, 4.4461897016e-01],
                     [9.2428863049e-01, 1.8203070760e-01, 9.6800571680e-01],
                     [2.0590199530e-01, 4.0901741385e-01, 6.0365653038e-01],
                     [5.5193322897e-01, 8.2984429598e-01, 5.1535344124e-01],
                     [6.6078317165e-01, 9.4258475304e-01, 6.4569509029e-01],
                     [6.1082202196e-01, 4.7290626168e-01, 4.6968898177e-01],
                     [4.1167530417e-01, 9.5446007326e-03, 1.1851237714e-01],
                     [3.5717773438e-01, 6.9824218750e-02, 6.5612792969e-02],
                     [2.2991758585e-01, 1.3940264285e-01, 9.9633944035e-01]],

            'OBSTACLE': [
                [2, 0, 1, 2],
                [2, 4, 1, 2],
                [2, 8, 1, 2],
                [2, 12, 1, 2],
                [7, 0, 2, 6],
                [7, 10, 1, 6],
                [13, 0, 1, 2],
                [13, 4, 1, 2],
                [13, 8, 1, 2],
                [13, 12, 1, 2],
            ],

            'STATION': [
                [8 / 16, 8 / 16],
                [5 / 16, 13 / 16],
                [11 / 16, 13 / 16],
                [11 / 16, 3 / 16],
                [5 / 16, 3 / 16]
            ],
            'CHANNEL': 3,

            'NUM_UAV': 2,
            'INIT_POSITION': (0, 8, 8),  # todo
            'MAX_ENERGY': 30.,  # must face the time of lack
            'NUM_ACTION': 2,  # 2
            'SAFE_ENERGY_RATE': 0.2,
            'RANGE': 1.0,
            'MAXDISTANCE': 1.,
            'COLLECTION_PROPORTION': 0.2,  # c speed
            'FILL_PROPORTION': 0.2,  # fill speed

            'WALL_REWARD': -1.,
            'VISIT': 1. / 1000.,
            'DATA_REWARD': 1.,
            'FILL_REWARD': 1.,
            'ALPHA': 1.,
            'BETA': 0.1,
            'EPSILON': 1e-4,
            'NORMALIZE': .1,
            'FACTOR': 0.1,
            'DiscreteToContinuous': [
                [-1.0, -1.0],
                [-1.0, -0.5],
                [-1.0, 0.0],
                [-1.0, 0.5],
                [-1.0, 1.0],
                [-0.5, -1.0],
                [-0.5, -0.5],
                [-0.5, 0.0],
                [-0.5, 0.5],
                [-0.5, 1.0],
                [0.0, -1.0],
                [0.0, -0.5],
                [0.0, 0.0],  # 重点关注no-op
                [0.0, 0.5],
                [0.0, 1.0],
                [0.5, -1.0],
                [0.5, -0.5],
                [0.5, 0.0],
                [0.5, 0.5],
                [0.5, 1.0],
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],

            ]
        }
        self.LOG = log
        if self.LOG is not None:
            self.time = log.time

    def log(self):
        if self.LOG is not None:
            self.LOG.log(self.V)
        else:
            pass