import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, sph_harm_y
from math import factorial
from matplotlib import cm

def enter_quantum_numbers():
    print("\/" * 15)
    print("Hydrogen atom visualizer")
    print("\/" * 15)

    while True:
        try:
            n = int(input("Principal quantum number  n:"))
            if n < 1:
                raise ValueError
        except ValueError:
            print("n must be a positive integer")
            continue

        try:
            l = int(input(f"orbital quantum number  l:"))
            if not (0 <= l <= n - 1):
                raise ValueError
        except ValueError:
            print(f"l must satisfy 0 ≤ l ≤ {n-1}.")
            continue

        try:
            m = int(input(f"Magnetic quantum number m: "))
            if not (-l <= m <= l):
                raise ValueError
        except ValueError:
            print(f"m must satisfy -{l} ≤ m ≤ {l}.")
            continue

        try:
            mode = int(input("choose mode: 2=probability density visualizer in 2D, 3=spherical harmonics visualizer in 3D:"))
            if mode not in (2, 3):  
                raise ValueError
        except ValueError:
            print("mode must be either 2 or 3")
            continue

        return n, l, m, mode


a0 = 1.0  #ett atomic constants to 1
def radial_wavefunction(n, l, r): 
    #Rn,l(r) proportional to Ln,l
    rho = 2.0 * r / (n * a0)   
    norm = np.sqrt(
        (2.0 / (n * a0)) ** 3
        * factorial(n - l - 1)
        / (2.0 * n * factorial(n + l))
    )
    L = genlaguerre(n - l - 1, 2 * l + 1)
    return norm * np.exp(-rho / 2.0) * rho ** l * L(rho)


def real_sph_harm(l, m, theta, phi): 
    #atomic orbital plotted with the real part of Yl,m 
    am = abs(m)

    if m == 0:
        return np.real(sph_harm_y(l, 0, theta, phi))
    elif m > 0:
        return np.real(
            (1.0 / np.sqrt(2))
            * ((-1) ** m * sph_harm_y(l, am, theta, phi) + sph_harm_y(l, -am, theta, phi))
        )
    else:
        return np.real(
            (1j / np.sqrt(2))
            * (-1*(-1) ** m * sph_harm_y(l, am, theta, phi) + sph_harm_y(l, -am, theta, phi))
        )


def hydrogen_prob_density(n, l, m, x, z):
    #P(r)=|psi_n,l,m|**2
    r = np.sqrt(x ** 2 + z ** 2)
    r = np.where(r < 1e-12, 1e-12, r)

    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi   = np.where(x >= 0, 0.0, np.pi)

    R = radial_wavefunction(n, l, r)
    Y = real_sph_harm(l, m, theta, phi)
    return (R * Y) ** 2


def plot_hydrogen_prob_density(n, l, m):
    r_ext = max(6, 2 * n) * n * a0
    grid_size = 1000
    grid = np.linspace(-r_ext, r_ext, grid_size)
    xgrid, zgrid = np.meshgrid(grid, grid)

    density = hydrogen_prob_density(n, l, m, xgrid, zgrid)

    pmax = np.percentile(density, 99.999999999999999)
    pmax = max(pmax, 1e-30)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_facecolor("white")

    img = ax.imshow(
        density,
        extent=[-r_ext, r_ext, -r_ext, r_ext],
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=pmax,
        interpolation="bilinear",
    )

    ax.set_title(
        f"Hydrogen Wave Function for \n$\\psi_{{{n},{l},{m}}}$ — xz-plane",
        color="black", fontsize=14, pad=12,
    )

    plt.show()

while True:
    n, l, m, mode = enter_quantum_numbers()

    if mode == 2:
        #show 2D plotting of the probability density of hydrogen atom
        print(f"\nRendering |ψ_{{{n},{l},{m}}}|²  …\n")  
        plot_hydrogen_prob_density(n, l, m)                             

    if mode == 3:
        #show 3D plotting for Ylm corresponding to the entered l,m regardless the entered  n
        theta = np.linspace(0, np.pi, 300)
        phi = np.linspace(0, 2 * np.pi, 300)
        thetagrid, phigrid = np.meshgrid(theta, phi)

        Y = sph_harm_y(l, m, thetagrid, phigrid)
        r = np.abs(Y)
        r = 2 * r + 0.5

        x_harm = r * np.sin(thetagrid) * np.cos(phigrid)
        y_harm = r * np.sin(thetagrid) * np.sin(phigrid)
        z_harm = r * np.cos(thetagrid)

        print (f"rendering Yl,m for l={l} , m={m} ")

        facecolors = cm.gist_heat(1 - (r - r.min()) / (r.max() - r.min() + 1e-10))

        fig = plt.figure(figsize=(8, 8))                   
        ax = fig.add_subplot(111, projection="3d")    
        ax.plot_surface(x_harm, y_harm, z_harm, facecolors=facecolors, linewidth=0)
        ax.set_title(f"Spherical Harmonic  l = {l},  m = {m}", fontsize=13)
    
        plt.show()
        plt.close()

    again = input("\nDo you want another run? (y/n): ")
    if again.lower() != "y":
        print("Exiting program...")
        break