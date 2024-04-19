import math
import hashlib
import matplotlib.pyplot as plt

def kompresiraj(ST):
    """Kompresira zaporedje bajtov ST v kompresijsko kodo KT."""

    # inicializacija slovarja PT - vsi veljavni 8-bitni znaki
    dict_size = 256
    PT = {bytes([i]): i for i in range(dict_size)}

    KT = []

    # 1. korak:
    s = bytes([ST[0]])

    # 2. korak:
    for i in range(1, len(ST)):
        t = bytes([ST[i]])
        u = s + t
        if u in PT:
            s = u
        else:
            KT.append(PT[s])
            PT[u] = dict_size
            dict_size += 1
            s = t

    # 3. korak:
    KT.append(PT[s])
 
    return KT
 
def dekompresiraj(KT):  # dobimo na vhod slovar
    """Dekompresira kompresijsko kodo KT v zaporedje bajtov ST."""
    # gradimo inverzni slovar PT - vsi veljavni 8-bitni znaki
    dict_size = 256
    PT = {i: bytes([i]) for i in range(dict_size)}

    # Pripravimo izhodno zaporedje ST
    ST = []

    # Preberemo prvi element iz KT
    s = PT[KT[0]]
    ST.append(s)

    for k in KT[1:]:    # for loop skozi KT slovar
        if k in PT:     # za vsak element iz KT preverimo, če je v PT
            entry = PT[k]       # če je v slovarju
        else:
            # če koda ni v slovarju, moramo uporabiti poseben primer
            entry = s + s[:1]  # s[:1] je prvi bajt v nizu s

        ST.append(entry)

        # Dodaj novo sekvenco v slovar
        PT[dict_size] = s + entry[:1]
        dict_size += 1

        # Posodobi s na trenutno entry
        s = entry

    # Spojimo bajte v končni rezultat
    return b''.join(ST)

def izracunaj_velikost(KT):
    """Izračuna velikost kompresiranega sporočila (KT), če bi ga zakodirali v bajtih."""
    if not KT:
        return 0

    # Največji element v KT za določanje bitov potrebnih za kodiranje
    max_k = max(KT)
    bitov_na_kodo = math.ceil(math.log2(max_k + 1))
    bajtov_na_kodo = math.ceil(bitov_na_kodo / 8)

    # Skupna velikost v bajtih
    total_bajtov = len(KT) * bajtov_na_kodo
    return total_bajtov

def preizkusi_kompresijo(datoteka):
    print("\n\nPreizkus gospodarnosti kodiranja na besedilni datoteki:")

    # Branje vsebine datoteke
    try:
        with open(datoteka, "rb") as f:
            bajti = f.read()
    except IOError:
        print(f"Napaka pri branju datoteke {datoteka}")
        return

    # Kompresiranje in dekompresiranje prebrane vsebine
    kompresirani_bajti = kompresiraj(bajti)
    dekompresirani_bajti = dekompresiraj(kompresirani_bajti)

    # Shranjevanje dekompresiranih bajtov nazaj v datoteko
    try:
        with open("dekompresirano.txt", "wb") as f:
            f.write(dekompresirani_bajti)
    except IOError:
        print("Napaka pri zapisovanju dekompresirane datoteke.")
        return

    # Izračun in primerjava MD5 hash-a originalne in dekompresirane datoteke
    hash_kom = hashlib.md5(bajti).hexdigest()
    hash_dekom = hashlib.md5(dekompresirani_bajti).hexdigest()
    print(f"MD5 hash originalne datoteke: {hash_kom}")
    print(f"MD5 hash dekompresirane datoteke: {hash_dekom}")
    print(f"Hasha originalne in dekompresirane datoteke {'sta enaka' if hash_kom == hash_dekom else 'nista enaka'}.")

# Zaženi preizkus
preizkusi_kompresijo("besedilo.txt")

if __name__ == "__main__":
    bajti = b"TRALALALALA"
    KT = kompresiraj(bajti)
    print(KT)
    ST = dekompresiraj(KT)
    print(ST)

    # preiskus na besedilni datoteki


    print("\n\nUspešnost kompresije na različnih dolžinah besedilne datoteke:")
    with open("besedilo.txt", "rb") as f:
        originalni_bajti = f.read()

    uspesnost = []
    dolzina = range(20000, 1038950, 20000)
    for bajti in dolzina:
        kompresirani_bajti = kompresiraj(originalni_bajti[:bajti])
        komp_velikost = izracunaj_velikost(kompresirani_bajti)
        uspesnost.append(komp_velikost / bajti)
        if bajti % 100000 == 0:
            print(f"Če vzamemo prvih {bajti} bajtov, je uspešnost kompresije {komp_velikost/bajti:.4f}")

    # izris grafa uspešnosti kompresije s pomočjo knjižnice matplotlib
    plt.plot(dolzina, uspesnost, "o-", label="Uspešnost kompresije")
    plt.title("Uspešnost kompresije na različnih dolžinah besedilne datoteke")
    plt.xlabel("Dolžina besedila [byte]")
    plt.ylabel("Uspešnost kompresije")
    plt.legend()
    plt.grid()
    plt.show()
