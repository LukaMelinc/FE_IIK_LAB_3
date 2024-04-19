import numpy as np
import hashlib
import os
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
 
def dekompresiraj(KT): 
    """Dekompresira kompresijsko kodo KT v zaporedje bajtov ST."""
    # gradimo inverzni slovar PT - vsi veljavni 8-bitni znaki
    dict_size = 256
    PT = {i: bytes([i]) for i in range(dict_size)}

    ST = []

    # TODO: implementiraj dekompresijo po postopku LZW
    s = bytes([KT.pop(0)])
    ST.append(s)
    for k in KT:
        if k in PT:
            entry = PT[k]
        elif k == dict_size:
            entry = s + bytes([s[0]])
        else:
            raise ValueError("Neveljavna koda v stiskanem sporočilu")
        ST.append(entry)
        PT[dict_size] = s + bytes([entry[0]])
        dict_size += 1
        s = entry

    # raise NotImplementedError
    rezultat = bytes()
    for element in ST:
        rezultat += element
    return rezultat


def izracunaj_velikost(KT):
    # TODO: implementiraj funkcijo, ki izračuna velikost kompresiranega
    #       sporočila (KT), če bi ga zakodirali v bajtih
    stevilo_bitov = 0
    for cifra in KT:
        if cifra.bit_length() < 8:
            stevilo_bitov += 8
        else:
            stevilo_bitov += cifra.bit_length()

    return int(np.ceil(stevilo_bitov/8))


if __name__ == "__main__":
    bajti = b"TRALALALALA"
    KT = kompresiraj(bajti)
    print(KT)
    ST = dekompresiraj(KT)
    print(ST.decode())

    # TODO 1: preberi datoteko besedilo.txt, jo kompresiraj, dekompresiraj in primerjaj MD5 hash datotek
    print("\n\nPreizkus gospodarnosti kodiranja na besedilni datoteki:")
    with open("besedilo.txt", "rb") as f:
        bajti = f.read()
    kompresirani_bajti = kompresiraj(bajti)
    dekompresirani_bajti = dekompresiraj(kompresirani_bajti)
    with open("dekompresirano.txt", "wb") as f:
        f.write(dekompresirani_bajti)

    hash_kom = hashlib.md5(bajti).hexdigest()
    hash_dekom = hashlib.md5(dekompresirani_bajti).hexdigest()
    print(f"Hash kompresirane datoteke: {hash_kom}")
    print(f"Hash dekompresirane datoteke: {hash_dekom}")
    print(f"Hasa kompresirane in dekompresirane datoteke {'sta' if hash_kom == hash_dekom else 'nista'} enaka.")

    # TODO 2: Preizkus gospodarnosti kodiranja na različnih vrstah datotek
    print("\n\nUspešnost kompresije na različnih vrstah datotek:")
    os.remove("kompresija_datotek.csv") if os.path.exists("kompresija_datotek.csv") else None
    for datoteka in os.listdir("Dodatno_gradivo"):
        with open(f"Dodatno_gradivo/{datoteka}", "rb") as f:
            bajti = f.read()
        kompresirani_bajti = kompresiraj(bajti)
        orig_velikost = os.path.getsize(f"Dodatno_gradivo/{datoteka}")
        komp_velikost = izracunaj_velikost(kompresirani_bajti)
        uspesnost = komp_velikost/orig_velikost  # uspešnost kompresije je razmerje med velikostjo datoteke pred in po
        print(f"{datoteka}, velikost:{orig_velikost}, kompresirana velikost:{komp_velikost}, uspesnost:{uspesnost:.4f}")
        with open("kompresija_datotek.csv", "a") as f:
            f.write(f"{datoteka},{orig_velikost},{komp_velikost},{uspesnost}\n")

    # TODO 3: preizkus gospodarnosti kodiranja na različnih dolžinah besedilne datoteke
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
