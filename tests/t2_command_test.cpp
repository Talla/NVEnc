// MCWiener canonical/alias parser and serialization roundtrip. used_by: native port receipt.
// rules: actual shared parser and serializer, no GPU, explicit error cases.
#include "../NVEncCore/rgy_cmd.h"
#include <iostream>
#include <stdexcept>
static void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
int main(){try{int cases=0;for(const TCHAR* option:{_T("vpp-mcwiener"),_T("vpp-t2")})for(const TCHAR* mode:{_T("plain"),_T("box"),_T("full")}){RGYParamVpp v,def;sArgsData args;tstring opt=tstring(_T("--"))+option,value=tstring(_T("mode="))+mode+_T(",profile=C:/profile with spaces/model.json,pp=SIM,iso=fixture");const TCHAR* input[]={opt.c_str(),value.c_str()};int i=0;require(parse_one_vpp_option(option,input,i,2,&v,&args)==0,"parse canonical/alias");auto cmd=gen_cmd(&v,&def,false);require(cmd.find(_T("--vpp-mcwiener"))!=tstring::npos,"canonical serialization");auto split=splitCommandLine(cmd.c_str());std::vector<const TCHAR*> argv;for(auto& s:split)if(!s.empty())argv.push_back(s.c_str());RGYParamVpp back;int j=0;require(argv.size()==2,"quoted single compound argument");require(parse_one_vpp_option(_T("vpp-mcwiener"),argv.data(),j,int(argv.size()),&back,&args)==0,"reparse serialized");require(back.t2==v.t2,"parameter equality after roundtrip");++cases;}
    for(const TCHAR* value:{_T("profile=a,pp=x,iso=y"),_T("mode=bad,profile=a,pp=x,iso=y"),_T("mode=box,pp=x,iso=y"),_T("mode=box,profile=a,iso=y"),_T("mode=box,profile=a,pp=x"),_T("mode=box,profile=a,profile=b,pp=x,iso=y"),_T("mode=box,profile=a,pp=x,iso=y,unexpected=z")}){RGYParamVpp v;sArgsData args;const TCHAR* argv[]={_T("--vpp-mcwiener"),value};int i=0;require(parse_one_vpp_option(_T("vpp-mcwiener"),argv,i,2,&v,&args)!=0,"malformed option rejected");++cases;}
    std::cout<<"{\"result\":\"PASS\",\"cases\":"<<cases<<",\"roundtrip\":6,\"negative\":7}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
