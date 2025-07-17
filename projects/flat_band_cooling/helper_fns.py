def get_model_str(lat_str, lat_dim, sys_len, V_rsv, tnnn = None):
    """Return a string that describes the model."""
    other_params_str = ""
    if lat_str == "kagome_withD":
        other_params_str += f"_tnnn{tnnn:.4f}"
    return (f"{lat_str}_lat{lat_dim[0]}x{lat_dim[1]}"
            f"_sys{sys_len}x{sys_len}_Vrsv{V_rsv:.4f}{other_params_str}").replace(".", "p")