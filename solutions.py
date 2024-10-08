puzzle_1 = "i->i"
puzzle_2 = "ijkl->ijkl"
puzzle_3 = "i,i->i"
puzzle_4 = "ijk,ijk->ijk"
puzzle_5 = "i->"
puzzle_6 = "i,i->i"
puzzle_7 = "i,i->"
puzzle_8 = "ij->i"
puzzle_9 = "ij,ij->j"
puzzle_10 = "ijk->ik"
puzzle_11 = "ii->i"
puzzle_12 = "ii->"
puzzle_13 = "ij,j->i"
puzzle_14 = "ijk,jk->i"
puzzle_15 = "i,j->ij"
puzzle_16 = "i,jk->ijk"
puzzle_17 = "ij->ji"
puzzle_18 = "ijk->kji"
puzzle_19 = "ijkl->ljik"
puzzle_20 = "ijk,ijk->kij"
puzzle_21 = "ij,jk->ik"
puzzle_22 = "ij,jk->ki"
puzzle_23 = "ij,kj->ik"
puzzle_24 = "ija,ak->ijk"
puzzle_25 = "ija,iak->ijk"
puzzle_26 = "iaj,ak->ijk"
puzzle_27 = "aij,ka->jik"
puzzle_28 = "iaj,akl->li"
puzzle_29 = "...ij->...ji"
puzzle_30 = "...ij,jk->...ik"


# Puzzle #31
def head(input_bld):
  q_blk = np.einsum("bld,dk->blk", input_bld, w_q_dk)
  k_blk = np.einsum("bld,dk->blk", input_bld, w_k_dk)
  v_blk = np.einsum("bld,dk->blk", input_bld, w_v_dk)
  scores_bll = np.einsum("bik,bjk->bij", q_blk, k_blk)
  scores_bll = softmax((scores_bll + mask) * k**-0.5)
  out_blk = np.einsum("blj,bjk->blk", scores_bll, v_blk)
  return out_blk


# Puzzle #32
def attention(input_bld):
  q_blhk = np.einsum("bld,dhk->blhk", input_bld, w_q_dhk)
  k_blhk = np.einsum("bld,dhk->blhk", input_bld, w_k_dhk)
  v_blhk = np.einsum("bld,dhk->blhk", input_bld, w_v_dhk)
  scores_bhll = np.einsum("bihk,bjhk->bhij", q_blhk, k_blhk)
  scores_bhll = softmax((scores_bhll + mask) * k**-0.5)
  out_blhk = np.einsum("bhli,bihk->blhk", scores_bhll, v_blhk)
  out_bld = np.einsum("blhk,hkd->bld", out_blhk, w_o_hkd)
  return out_bld
