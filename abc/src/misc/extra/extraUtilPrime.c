/**CFile****************************************************************

  FileName    [extraUtilPrime.c]

  SystemName  [ABC: Logic synthesis and verification system.]

  PackageName [extra]

  Synopsis    [Function enumeration.]

  Author      [Alan Mishchenko]
  
  Affiliation [UC Berkeley]

  Date        [Ver. 1.0. Started - June 20, 2005.]

  Revision    [$Id: extraUtilPrime.c,v 1.0 2003/02/01 00:00:00 alanmi Exp $]

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "misc/vec/vec.h"
#include "misc/vec/vecHsh.h"
#include "bool/kit/kit.h"
#include "misc/extra/extra.h"

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    []

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Abc_GenCountDump( Vec_Int_t * vPrimes, int nVars, char * pFileName )
{
    FILE * pFile;
    int i, k, Prime;
    pFile = fopen( pFileName, "wb" );
    fprintf( pFile, "# %d prime numbers up to 2^%d generated by ABC on %s\n", Vec_IntSize(vPrimes), nVars, Extra_TimeStamp() );
    fprintf( pFile, ".i %d\n", nVars );
    fprintf( pFile, ".o %d\n", 1 );
    fprintf( pFile, ".p %d\n", Vec_IntSize(vPrimes) );
    Vec_IntForEachEntry( vPrimes, Prime, i )
        for ( k = nVars-1; k >= 0; k-- )
            fprintf( pFile, "%d%s", (Prime >> k)&1, k ? "" : " 1\n" );
    fprintf( pFile, ".e\n\n" );
    fclose( pFile );
}

/**Function*************************************************************

  Synopsis    []

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Abc_GenCountHits1( Vec_Bit_t * vMap, Vec_Int_t * vPrimes, int nVars )
{
    abctime clk = Abc_Clock();
    int i, k, Prime, Count = 0;
    Vec_IntForEachEntry( vPrimes, Prime, i )
    {
        for ( k = 0; k < nVars; k++ )
            if ( !Vec_BitEntry(vMap, Prime ^ (1<<k)) )
            {
                //printf( "%3d : %2d %2d     flipped bit %d\n", Count, Prime, Prime ^ (1<<k), k );
                Count++;
            }
    }
    printf( "Dist1 pairs = %d. ", Count/2 );
    Abc_PrintTime( 1, "Time", Abc_Clock() - clk );
}
Vec_Int_t * Abc_GenPrimes( int nVars )
{
    int i, n, nBits = ( 1 << nVars );
    Vec_Bit_t * vMap = Vec_BitStart( nBits );
    Vec_Int_t * vPrimes = Vec_IntAlloc( 1000 );
    Vec_BitWriteEntry(vMap, 0, 1);
    Vec_BitWriteEntry(vMap, 1, 1);
    for ( n = 2; n < nBits; n++ )
        if ( !Vec_BitEntry(vMap, n) )
            for ( i = 2*n; i < nBits; i += n )
                Vec_BitWriteEntry(vMap, i, 1);
    for ( n = 2; n < nBits; n++ )
        if ( !Vec_BitEntry(vMap, n) )
            Vec_IntPush( vPrimes, n );
    printf( "Primes up to 2^%d = %d\n", nVars, Vec_IntSize(vPrimes) );
    Abc_GenCountHits1( vMap, vPrimes, nVars );
    Vec_BitFree( vMap );
    return vPrimes;
}
void Abc_GenPrimesTest()
{
    // 54,400,028 primes up to 2^30 can be computed in 22 sec
    int nVars = 18;
    Vec_Int_t * vPrimes = Abc_GenPrimes( nVars );
    Abc_GenCountDump( vPrimes, nVars, "primes18.pla" );
    //Vec_IntPrint( vPrimes );
    printf( "Primes up to 2^%d = %d\n", nVars, Vec_IntSize(vPrimes) );

    Vec_IntFree( vPrimes );
}




#define ABC_PRIME_MASK 0xFF
static unsigned s_256Primes[ABC_PRIME_MASK+1] = 
{
    0x984b6ad9,0x18a6eed3,0x950353e2,0x6222f6eb,0xdfbedd47,0xef0f9023,0xac932a26,0x590eaf55,
    0x97d0a034,0xdc36cd2e,0x22736b37,0xdc9066b0,0x2eb2f98b,0x5d9c7baf,0x85747c9e,0x8aca1055,
    0x50d66b74,0x2f01ae9e,0xa1a80123,0x3e1ce2dc,0xebedbc57,0x4e68bc34,0x855ee0cf,0x17275120,
    0x2ae7f2df,0xf71039eb,0x7c283eec,0x70cd1137,0x7cf651f3,0xa87bfa7a,0x14d87f02,0xe82e197d,
    0x8d8a5ebe,0x1e6a15dc,0x197d49db,0x5bab9c89,0x4b55dea7,0x55dede49,0x9a6a8080,0xe5e51035,
    0xe148d658,0x8a17eb3b,0xe22e4b38,0xe5be2a9a,0xbe938cbb,0x3b981069,0x7f9c0c8e,0xf756df10,
    0x8fa783f7,0x252062ce,0x3dc46b4b,0xf70f6432,0x3f378276,0x44b137a1,0x2bf74b77,0x04892ed6,
    0xfd318de1,0xd58c235e,0x94c6d25b,0x7aa5f218,0x35c9e921,0x5732fbbb,0x06026481,0xf584a44f,
    0x946e1b5f,0x8463d5b2,0x4ebca7b2,0x54887b15,0x08d1e804,0x5b22067d,0x794580f6,0xb351ea43,
    0xbce555b9,0x19ae2194,0xd32f1396,0x6fc1a7f1,0x1fd8a867,0x3a89fdb0,0xea49c61c,0x25f8a879,
    0xde1e6437,0x7c74afca,0x8ba63e50,0xb1572074,0xe4655092,0xdb6f8b1c,0xc2955f3c,0x327f85ba,
    0x60a17021,0x95bd261d,0xdea94f28,0x04528b65,0xbe0109cc,0x26dd5688,0x6ab2729d,0xc4f029ce,
    0xacf7a0be,0x4c912f55,0x34c06e65,0x4fbb938e,0x1533fb5f,0x03da06bd,0x48262889,0xc2523d7d,
    0x28a71d57,0x89f9713a,0xf574c551,0x7a99deb5,0x52834d91,0x5a6f4484,0xc67ba946,0x13ae698f,
    0x3e390f34,0x34fc9593,0x894c7932,0x6cf414a3,0xdb7928ab,0x13a3b8a3,0x4b381c1d,0xa10b54cb,
    0x55359d9d,0x35a3422a,0x58d1b551,0x0fd4de20,0x199eb3f4,0x167e09e2,0x3ee6a956,0x5371a7fa,
    0xd424efda,0x74f521c5,0xcb899ff6,0x4a42e4f4,0x747917b6,0x4b08df0b,0x090c7a39,0x11e909e4,
    0x258e2e32,0xd9fad92d,0x48fe5f69,0x0545cde6,0x55937b37,0x9b4ae4e4,0x1332b40e,0xc3792351,
    0xaff982ef,0x4dba132a,0x38b81ef1,0x28e641bf,0x227208c1,0xec4bbe37,0xc4e1821c,0x512c9d09,
    0xdaef1257,0xb63e7784,0x043e04d7,0x9c2cea47,0x45a0e59a,0x281315ca,0x849f0aac,0xa4071ed3,
    0x0ef707b3,0xfe8dac02,0x12173864,0x471f6d46,0x24a53c0a,0x35ab9265,0xbbf77406,0xa2144e79,
    0xb39a884a,0x0baf5b6d,0xcccee3dd,0x12c77584,0x2907325b,0xfd1adcd2,0xd16ee972,0x345ad6c1,
    0x315ebe66,0xc7ad2b8d,0x99e82c8d,0xe52da8c8,0xba50f1d3,0x66689cd8,0x2e8e9138,0x43e15e74,
    0xf1ced14d,0x188ec52a,0xe0ef3cbb,0xa958aedc,0x4107a1bc,0x5a9e7a3e,0x3bde939f,0xb5b28d5a,
    0x596fe848,0xe85ad00c,0x0b6b3aae,0x44503086,0x25b5695c,0xc0c31dcd,0x5ee617f0,0x74d40c3a,
    0xd2cb2b9f,0x1e19f5fa,0x81e24faf,0xa01ed68f,0xcee172fc,0x7fdf2e4d,0x002f4774,0x664f82dd,
    0xc569c39a,0xa2d4dcbe,0xaadea306,0xa4c947bf,0xa413e4e3,0x81fb5486,0x8a404970,0x752c980c,
    0x98d1d881,0x5c932c1e,0xeee65dfb,0x37592cdd,0x0fd4e65b,0xad1d383f,0x62a1452f,0x8872f68d,
    0xb58c919b,0x345c8ee3,0xb583a6d6,0x43d72cb3,0x77aaa0aa,0xeb508242,0xf2db64f8,0x86294328,
    0x82211731,0x1239a9d5,0x673ba5de,0xaf4af007,0x44203b19,0x2399d955,0xa175cd12,0x595928a7,
    0x6918928b,0xde3126bb,0x6c99835c,0x63ba1fa2,0xdebbdff0,0x3d02e541,0xd6f7aac6,0xe80b4cd0,
    0xd0fa29f1,0x804cac5e,0x2c226798,0x462f624c,0xad05b377,0x22924fcd,0xfbea205c,0x1b47586d
};



#define TAB_UNUSED 0xFFFF

typedef struct Tab_Man_t_ Tab_Man_t;
typedef struct Tab_Ent_t_ Tab_Ent_t;
struct Tab_Man_t_
{
    int         nVars;
    int         nCubes;
    int         nLits; 
    int         nTable;
    int *       pCubes;   // pointers to cubes
    word *      pValues;  // hash values
    Tab_Ent_t * pTable;   // hash table (lits -> cube + lit + lit)
    int         Degree;   // degree of 2 larger than log2(nCubes)
    int         Mask;     // table size (2^Degree)
    int         nEnts;    // number of entries
};
struct Tab_Ent_t_
{
    int         Table;
    int         Cube;
    unsigned    VarA : 16;
    unsigned    VarB : 16;
    int         Next;
};

static inline int *       Tab_ManCube( Tab_Man_t * p, int i ) { assert(i >= 0  && i < p->nCubes); return p->pCubes + i * (p->nVars + 1);  }
static inline Tab_Ent_t * Tab_ManEnt( Tab_Man_t * p, int i )  { assert(i >= -1 && i < p->nTable); return i >= 0 ? p->pTable + i : NULL;   }

static inline int   Tab_ManValue( Tab_Man_t * p, int a )
{
    assert( a >= 0 && a < 256 );
    return s_256Primes[a];
}
static inline int   Tab_ManFinal( Tab_Man_t * p, int a )
{
    return a & p->Mask;
}
static inline word Tab_ManHashValue( Tab_Man_t * p, int * pCube )
{
    word Value = 0; int i;
    for ( i = 1; i <= pCube[0]; i++ )
        Value += Tab_ManValue( p, pCube[i] );
    return Value;
}
static inline word Tab_ManHashValueWithoutVar( Tab_Man_t * p, int * pCube, int iVar )
{
    word Value = 0; int i;
    for ( i = 1; i <= pCube[0]; i++ )
        if ( i != iVar )
            Value += Tab_ManValue( p, pCube[i] );
    return Value;
}
static inline unsigned Tab_ManHashValueCube( Tab_Man_t * p, int c, int iVar )
{
    if ( iVar == 0xFFFF )
        return (unsigned)(p->pValues[c] % p->nTable);
    return (unsigned)((p->pValues[c] - Tab_ManValue(p, Tab_ManCube(p, c)[iVar+1])) % p->nTable);
}
static inline void  Tab_ManPrintCube( Tab_Man_t * p, int c, int Var )
{
    int i, * pCube = Tab_ManCube( p, c );
    for ( i = 1; i <= pCube[0]; i++ )
//        if ( i == Var + 1 )
//            printf( "-" );
//        else
            printf( "%d", !Abc_LitIsCompl(pCube[i]) );
}
static inline void  Tab_ManHashAdd( Tab_Man_t * p, int Value, int Cube, int VarA, int VarB )
{
    Tab_Ent_t * pCell = p->pTable + p->nEnts;
    Tab_Ent_t * pBin  = p->pTable + Value;
/*
    printf( "Adding cube " );
    Tab_ManPrintCube( p, Cube, VarA );
    printf( " with var %d and value %d\n", VarA, Value );
*/
    if ( pBin->Table >= 0 )
        pCell->Next = pBin->Table;
    pBin->Table = p->nEnts++;
    pCell->Cube = Cube;
    pCell->VarA = VarA;
    pCell->VarB = VarB;
}
static inline void  Tab_ManPrintEntry( Tab_Man_t * p, int e )
{
    printf( "Entry %10d : ", e );
    printf( "Cube %6d  ", p->pTable[e].Cube );
    printf( "Value %12u  ", Tab_ManHashValueCube(p, p->pTable[e].Cube, p->pTable[e].VarA) % p->nTable );
    Tab_ManPrintCube( p, p->pTable[e].Cube, p->pTable[e].VarA );
    printf( "   " );
    if ( p->pTable[e].VarA != 0xFFFF )
        printf( "%2d ", p->pTable[e].VarA );
    else
        printf( "   " );
    if ( p->pTable[e].VarB != 0xFFFF )
        printf( "%2d ", p->pTable[e].VarB );
    else
        printf( "   " );
    printf( "\n" );
} 
static inline void  Tab_ManHashCollectBin( Tab_Man_t * p, int Bin, Vec_Int_t * vBin )
{
    Tab_Ent_t * pEnt = p->pTable + Bin;
    Vec_IntClear( vBin );
    for ( pEnt = Tab_ManEnt(p, pEnt->Table); pEnt; pEnt = Tab_ManEnt(p, pEnt->Next) )
    {
        Vec_IntPush( vBin, pEnt - p->pTable );
        //Tab_ManPrintEntry( p, pEnt - p->pTable );
    }
    //printf( "\n" );
}

#define Tab_ManForEachCube( p, pCube, c )                                  \
    for ( c = 0; c < p->nCubes && (pCube = Tab_ManCube(p, c)); c++ )       \
        if ( pCube[0] == -1 ) {} else 

#define Tab_ManForEachCubeReverse( p, pCube, c )                           \
    for ( c = p->nCubes - 1; c >= 0 && (pCube = Tab_ManCube(p, c)); c-- )  \
        if ( pCube[0] == -1 ) {} else 


/**Function*************************************************************

  Synopsis    [Manager manipulation.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
Tab_Man_t * Tab_ManAlloc( int nVars, int nCubes )
{
    Tab_Man_t * p = ABC_CALLOC( Tab_Man_t, 1 );
    p->nVars   = nVars;
    p->nCubes  = nCubes;
    p->Degree  = Abc_Base2Log((p->nVars + 1) * p->nCubes + 1) + 3;
    p->Mask    = (1 << p->Degree) - 1;
    //p->nEnts   = 1;
    p->pCubes  = ABC_CALLOC( int, p->nCubes * (p->nVars + 1) );
    p->pValues = ABC_CALLOC( word, p->nCubes );
//    p->pTable  = ABC_CALLOC( Tab_Ent_t, (p->Mask + 1) );
    printf( "Allocated %.2f MB for cube structure.\n", 4.0 * p->nCubes * (p->nVars + 2) / (1 << 20) );
    return p;
}
void Tab_ManFree( Tab_Man_t * p )
{
    ABC_FREE( p->pCubes );
    ABC_FREE( p->pValues );
    ABC_FREE( p->pTable );
    ABC_FREE( p );
}
void Tab_ManStart( Tab_Man_t * p, Vec_Int_t * vCubes )
{
    int * pCube, Cube, c, v;
    p->nLits = 0;
    Tab_ManForEachCube( p, pCube, c )
    {
        Cube = Vec_IntEntry( vCubes, c );
        pCube[0] = p->nVars;
        for ( v = 0; v < p->nVars; v++ )
            pCube[v+1] = Abc_Var2Lit( v, !((Cube >> v) & 1) );
        p->pValues[c] = Tab_ManHashValue( p, pCube );
        p->nLits += pCube[0];
    }
}


/**Function*************************************************************

  Synopsis    [Find a cube-free divisor of the two cubes.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Tab_ManCubeFree( int * pCube1, int * pCube2, Vec_Int_t * vCubeFree )
{
    int * pBeg1 = pCube1 + 1;  // skip variable ID
    int * pBeg2 = pCube2 + 1;  // skip variable ID
    int * pEnd1 = pBeg1 + pCube1[0];
    int * pEnd2 = pBeg2 + pCube2[0];
    int Counter = 0, fAttr0 = 0, fAttr1 = 1;
    Vec_IntClear( vCubeFree );
    while ( pBeg1 < pEnd1 && pBeg2 < pEnd2 )
    {
        if ( *pBeg1 == *pBeg2 )
            pBeg1++, pBeg2++, Counter++;
        else if ( *pBeg1 < *pBeg2 )
            Vec_IntPush( vCubeFree, Abc_Var2Lit(*pBeg1++, fAttr0) );
        else  
        {
            if ( Vec_IntSize(vCubeFree) == 0 )
                fAttr0 = 1, fAttr1 = 0;
            Vec_IntPush( vCubeFree, Abc_Var2Lit(*pBeg2++, fAttr1) );
        }
    }
    while ( pBeg1 < pEnd1 )
        Vec_IntPush( vCubeFree, Abc_Var2Lit(*pBeg1++, fAttr0) );
    while ( pBeg2 < pEnd2 )
        Vec_IntPush( vCubeFree, Abc_Var2Lit(*pBeg2++, fAttr1) );
    if ( Vec_IntSize(vCubeFree) == 0 )
        printf( "The SOP has duplicated cubes.\n" );
    else if ( Vec_IntSize(vCubeFree) == 1 )
        printf( "The SOP has contained cubes.\n" );
//    else if ( Vec_IntSize(vCubeFree) == 2 && Abc_Lit2Var(Abc_Lit2Var(Vec_IntEntry(vCubeFree, 0))) == Abc_Lit2Var(Abc_Lit2Var(Vec_IntEntry(vCubeFree, 1))) )
//        printf( "The SOP has distance-1 cubes or it is not a prime cover.  Please make sure the result verifies.\n" );
    assert( !Abc_LitIsCompl(Vec_IntEntry(vCubeFree, 0)) );
    return Counter;
}
int Tab_ManCheckEqual2( int * pCube1, int * pCube2, int Var1, int Var2 )
{
    int i1, i2;
    for ( i1 = i2 = 1; ; i1++, i2++ )
    {
        if ( i1 == Var1 )  i1++;
        if ( i2 == Var2 )  i2++;
        if ( i1 > pCube1[0] || i2 > pCube2[0] )
            return 0;
        if ( pCube1[i1] != pCube2[i2] )
            return 0;
        if ( i1 == pCube1[0] && i2 == pCube2[0] )
            return 1;
    }
}
int Tab_ManCheckEqual( int * pCube1, int * pCube2, int Var1, int Var2 )
{
    int Cube1[32], Cube2[32];
    int i, k, nVars1, nVars2;
    assert( pCube1[0] <= 32 );
    assert( pCube2[0] <= 32 );
    for ( i = 1, k = 0; i <= pCube1[0]; i++ )
        if ( i != Var1 )
            Cube1[k++] = pCube1[i];
    nVars1 = k;
    for ( i = 1, k = 0; i <= pCube2[0]; i++ )
        if ( i != Var2 )
            Cube2[k++] = pCube2[i];
    nVars2 = k;
    if ( nVars1 != nVars2 )
        return 0;
    for ( i = 0; i < nVars1; i++ )
        if ( Cube1[i] != Cube2[i] )
            return 0;
    return 1;
}

/**Function*************************************************************

  Synopsis    [Collecting distance-1 pairs.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Tab_ManCountItems( Tab_Man_t * p, int Dist2, Vec_Int_t ** pvStarts )
{
    Vec_Int_t * vStarts = Vec_IntAlloc( p->nCubes );
    int * pCube, c, Count = 0;
    Tab_ManForEachCube( p, pCube, c )
    {
        Vec_IntPush( vStarts, Count );
        Count += 1 + pCube[0];
        if ( Dist2 )
            Count += pCube[0] * pCube[0] / 2;
    }
    assert( Vec_IntSize(vStarts) == p->nCubes );
    if ( pvStarts )
        *pvStarts = vStarts;
    return Count;
}
Vec_Int_t * Tab_ManCollectDist1( Tab_Man_t * p, int Dist2 )
{
    Vec_Int_t * vStarts = NULL;                           // starting mark for each cube
    int nItems = Tab_ManCountItems( p, Dist2, &vStarts ); // item count
    int nBits  = Abc_Base2Log( nItems ) + 6;              // hash table size  
    Vec_Bit_t * vPres = Vec_BitStart( 1 << nBits );       // hash table
    Vec_Bit_t * vMarks = Vec_BitStart( nItems );          // collisions 
    Vec_Int_t * vUseful = Vec_IntAlloc( 1000 );           // useful pairs
    Vec_Int_t * vBin = Vec_IntAlloc( 100 );
    Vec_Int_t * vCubeFree = Vec_IntAlloc( 100 );
    word Value; unsigned Mask = (1 << nBits) - 1;
    int * pCube, c, a, b, nMarks = 0, nUseful, Entry1, Entry2;
    // iterate forward
    Tab_ManForEachCube( p, pCube, c )
    {
        // cube
        if ( Vec_BitAddEntry(vPres, (int)p->pValues[c] & Mask) )
            Vec_BitWriteEntry( vMarks, nMarks, 1 );
        nMarks++;
        // dist1
        for ( a = 1; a <= pCube[0]; a++, nMarks++ )
            if ( Vec_BitAddEntry(vPres, (int)(p->pValues[c] - Tab_ManValue(p, pCube[a])) & Mask) )
                Vec_BitWriteEntry( vMarks, nMarks, 1 );
        // dist2
        if ( Dist2 )
        for ( a = 1; a <= pCube[0]; a++ )
        {
            Value = p->pValues[c] - Tab_ManValue(p, pCube[a]);
            for ( b = a + 1; b <= pCube[0]; b++, nMarks++ )
                if ( Vec_BitAddEntry(vPres, (int)(Value - Tab_ManValue(p, pCube[b])) & Mask) )
                    Vec_BitWriteEntry( vMarks, nMarks, 1 );
        }
    }
    assert( nMarks == nItems );
    Vec_BitReset( vPres );
    // iterate backward
    nMarks--;
    Tab_ManForEachCubeReverse( p, pCube, c )
    {
        Value = p->pValues[c];
        // dist2
        if ( Dist2 )
        for ( a = pCube[0]; a >= 1; a-- )
        {
            Value = p->pValues[c] - Tab_ManValue(p, pCube[a]);
            for ( b = pCube[0]; b >= a + 1; b--, nMarks-- )
                if ( Vec_BitAddEntry(vPres, (int)(Value - Tab_ManValue(p, pCube[b])) & Mask) )
                    Vec_BitWriteEntry( vMarks, nMarks, 1 );
        }
        // dist1
        for ( a = pCube[0]; a >= 1; a--, nMarks-- )
            if ( Vec_BitAddEntry(vPres, (int)(p->pValues[c] - Tab_ManValue(p, pCube[a])) & Mask) )
                Vec_BitWriteEntry( vMarks, nMarks, 1 );
        // cube
        if ( Vec_BitAddEntry(vPres, (int)p->pValues[c] & Mask) )
            Vec_BitWriteEntry( vMarks, nMarks, 1 );
        nMarks--;
    }
    nMarks++;
    assert( nMarks == 0 );
    Vec_BitFree( vPres );
    // count useful
    nUseful = Vec_BitCount( vMarks );
printf( "Items = %d. Bits = %d.   Useful = %d.   \n", nItems, nBits, nUseful );

    // add to the hash table
    p->nTable = Abc_PrimeCudd(nUseful);
    p->pTable = ABC_FALLOC( Tab_Ent_t, p->nTable );
printf( "Table %d\n", p->nTable );
    Tab_ManForEachCube( p, pCube, c )
    {
        // cube
        if ( Vec_BitEntry(vMarks, nMarks++) )
            Tab_ManHashAdd( p, (int)(p->pValues[c] % p->nTable), c, TAB_UNUSED, TAB_UNUSED );
        // dist1
        for ( a = 1; a <= pCube[0]; a++, nMarks++ )
            if ( Vec_BitEntry(vMarks, nMarks) )
                Tab_ManHashAdd( p, (int)((p->pValues[c] - Tab_ManValue(p, pCube[a])) % p->nTable), c, a-1, TAB_UNUSED );
        // dist2
        if ( Dist2 )
        for ( a = 1; a <= pCube[0]; a++ )
        {
            Value = p->pValues[c] - Tab_ManValue(p, pCube[a]);
            for ( b = a + 1; b <= pCube[0]; b++, nMarks++ )
                if ( Vec_BitEntry(vMarks, nMarks) )
                    Tab_ManHashAdd( p, (int)((Value - Tab_ManValue(p, pCube[b])) % p->nTable), c, a-1, b-1 );
        }
    }
    assert( nMarks == nItems );
    // collect entries
    for ( c = 0; c < p->nTable; c++ )
    {
        Tab_ManHashCollectBin( p, c, vBin );
        //printf( "%d ", Vec_IntSize(vBin) );
        //if ( c > 100 )
        //    break;
        Vec_IntForEachEntry( vBin, Entry1, a )
        Vec_IntForEachEntryStart( vBin, Entry2, b, a + 1 )
        {
            Tab_Ent_t * pEntA = Tab_ManEnt( p, Entry1 );
            Tab_Ent_t * pEntB = Tab_ManEnt( p, Entry2 );
            int * pCubeA = Tab_ManCube( p, pEntA->Cube );
            int * pCubeB = Tab_ManCube( p, pEntB->Cube );
//            int Base = Tab_ManCubeFree( pCubeA, pCubeB, vCubeFree );
//            if ( Vec_IntSize(vCubeFree) == 2 )
            if ( Tab_ManCheckEqual(pCubeA, pCubeB, pEntA->VarA+1, pEntB->VarA+1) )
            {
                Vec_IntPushTwo( vUseful, pEntA->Cube, pEntB->Cube );
            }
        }

    }
    //printf( "\n" );

    ABC_FREE( p->pTable );
    Vec_IntFree( vCubeFree );
    Vec_IntFree( vBin );
    Vec_BitFree( vMarks );
    return vUseful;
}

/**Function*************************************************************

  Synopsis    [Table decomposition.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Tab_DecomposeTest()
{
    int nVars = 20;// no more than 13
    abctime clk = Abc_Clock();
    Vec_Int_t * vPairs;
    Vec_Int_t * vPrimes = Abc_GenPrimes( nVars );
    Tab_Man_t * p = Tab_ManAlloc( nVars, Vec_IntSize(vPrimes) );
    Tab_ManStart( p, vPrimes );
    printf( "Created %d cubes dependent on %d variables with %d literals.\n", p->nCubes, p->nVars );
    vPairs = Tab_ManCollectDist1( p, 0 );
    printf( "Collected %d pairs.\n", Vec_IntSize(vPairs)/2 );
    Vec_IntFree( vPairs );
    Tab_ManFree( p );
    Vec_IntFree( vPrimes );
    Abc_PrintTime( 1, "Time", Abc_Clock() - clk );
}


////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////


ABC_NAMESPACE_IMPL_END

